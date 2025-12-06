import asyncio
import websockets
import json
import torch
import numpy as np
import cv2
import base64
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).parent.parent))

from training.model import ContrastiveModel
from inference.nearest_neighbor import EmbeddingIndex

# Braille pattern mapping
BRAILLE_PATTERNS = {
    (1,): 'a', (1,2): 'b', (1,4): 'c', (1,4,5): 'd', (1,5): 'e',
    (1,2,4): 'f', (1,2,4,5): 'g', (1,2,5): 'h', (2,4): 'i', (2,4,5): 'j',
    (1,3): 'k', (1,2,3): 'l', (1,3,4): 'm', (1,3,4,5): 'n', (1,3,5): 'o',
    (1,2,3,4): 'p', (1,2,3,4,5): 'q', (1,2,3,5): 'r', (2,3,4): 's',
    (2,3,4,5): 't', (1,3,6): 'u', (1,2,3,6): 'v', (2,4,5,6): 'w',
    (1,3,4,6): 'x', (1,3,4,5,6): 'y', (1,3,5,6): 'z', (): ' ',
}


class UnifiedInferenceServer:
    """Unified server for finger reading and Braille recognition."""
    
    def __init__(
        self,
        model_path: str = 'models/best_model.pt',
        index_path: str = 'models/inference_index.pkl',
        device: str = None
    ):
        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"🚀 Starting Unified Inference Server")
        print(f"Device: {self.device}")
        
        # Load finger reading model
        if Path(model_path).exists():
            print(f"Loading finger reading model from {model_path}...")
            self.model = ContrastiveModel(
                sensor_input_dim=5,
                audio_input_dim=768,
                embedding_dim=128
            )
            
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            print("✓ Finger reading model loaded")
            
            # Load index
            if Path(index_path).exists():
                print(f"Loading embedding index from {index_path}...")
                self.index = EmbeddingIndex.load(index_path, use_gpu=(self.device == 'cuda'))
                print("✓ Index loaded")
            else:
                print("⚠️  Index not found - finger reading will not work")
                self.index = None
        else:
            print("⚠️  Model not found - finger reading will not work")
            self.model = None
            self.index = None
        
        print(f"\n✅ Server ready!")
        print(f"   Finger reading: {'✓' if self.model and self.index else '✗'}")
        print(f"   Braille recognition: ✓")
    
    def preprocess_sensor_data(self, data: dict) -> torch.Tensor:
        """Preprocess sensor data for finger reading."""
        finger = np.array(data['finger'])
        imu = np.array(data['imu'])
        
        min_len = min(len(finger), len(imu))
        finger = finger[:min_len]
        imu = imu[:min_len]
        
        sensor = np.concatenate([finger, imu], axis=1)
        sensor = (sensor - sensor.mean(axis=0)) / (sensor.std(axis=0) + 1e-8)
        
        sensor_tensor = torch.tensor(sensor, dtype=torch.float32)
        sensor_tensor = sensor_tensor.unsqueeze(0)
        
        return sensor_tensor.to(self.device)
    
    def predict_finger(self, sensor_data: dict, top_k: int = 5) -> list:
        """Predict text from finger sensor data."""
        if not self.model or not self.index:
            return [{'text': 'Model not loaded', 'confidence': 0.0}]
        
        sensor_tensor = self.preprocess_sensor_data(sensor_data)
        
        with torch.no_grad():
            sensor_emb = self.model.sensor_encoder(sensor_tensor)
            sensor_emb = sensor_emb.cpu().numpy()[0]
        
        # Pad to match index dimension
        if sensor_emb.shape[0] < self.index.dimension:
            padding = np.zeros(self.index.dimension - sensor_emb.shape[0])
            sensor_emb = np.concatenate([sensor_emb, padding])
        
        results = self.index.search(sensor_emb, k=top_k)
        
        return [
            {'text': text, 'confidence': float(similarity)}
            for text, similarity, _ in results
        ]
    
    def detect_braille_dots(self, image: np.ndarray) -> list:
        """Detect Braille dots in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 500:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                circle_area = np.pi * radius * radius
                circularity = area / circle_area if circle_area > 0 else 0
                
                if circularity > 0.6:
                    dots.append((int(x), int(y)))
        
        return dots
    
    def cluster_dots_into_cells(self, dots: list, cell_width: int = 30, cell_height: int = 40) -> list:
        """Cluster dots into Braille cells."""
        if not dots:
            return []
        
        # Sort dots by x-coordinate
        dots_sorted = sorted(dots, key=lambda d: d[0])
        
        # Group into columns (cells)
        cells = []
        current_cell = []
        last_x = dots_sorted[0][0] if dots_sorted else 0
        
        for dot in dots_sorted:
            x, y = dot
            # New cell if gap > cell_width
            if x - last_x > cell_width:
                if current_cell:
                    cells.append(current_cell)
                current_cell = [dot]
            else:
                current_cell.append(dot)
            last_x = x
        
        if current_cell:
            cells.append(current_cell)
        
        return cells
    
    def dots_to_character(self, dots: list) -> str:
        """Convert dot positions to Braille character."""
        if not dots or len(dots) == 0:
            return ' '
        
        # Sort dots top-to-bottom, left-to-right
        dots_sorted = sorted(dots, key=lambda d: (d[1], d[0]))
        
        # Calculate centroid
        cx = sum(d[0] for d in dots_sorted) / len(dots_sorted)
        cy = sum(d[1] for d in dots_sorted) / len(dots_sorted)
        
        # Determine which Braille positions (1-6)
        positions = []
        for x, y in dots_sorted:
            # Left column (1,2,3) or right column (4,5,6)
            if x < cx:
                # Left column
                if y < cy - 10:
                    positions.append(1)
                elif y > cy + 10:
                    positions.append(3)
                else:
                    positions.append(2)
            else:
                # Right column
                if y < cy - 10:
                    positions.append(4)
                elif y > cy + 10:
                    positions.append(6)
                else:
                    positions.append(5)
        
        # Map to character
        positions_tuple = tuple(sorted(set(positions)))
        
        # Braille pattern mapping
        braille_map = {
            (1,): 'a', (1,2): 'b', (1,4): 'c', (1,4,5): 'd', (1,5): 'e',
            (1,2,4): 'f', (1,2,4,5): 'g', (1,2,5): 'h', (2,4): 'i', (2,4,5): 'j',
            (1,3): 'k', (1,2,3): 'l', (1,3,4): 'm', (1,3,4,5): 'n', (1,3,5): 'o',
            (1,2,3,4): 'p', (1,2,3,4,5): 'q', (1,2,3,5): 'r', (2,3,4): 's',
            (2,3,4,5): 't', (1,3,6): 'u', (1,2,3,6): 'v', (2,4,5,6): 'w',
            (1,3,4,6): 'x', (1,3,4,5,6): 'y', (1,3,5,6): 'z',
        }
        
        return braille_map.get(positions_tuple, '?')
    
    def recognize_braille(self, image_base64: str, finger_x: float = None, finger_y: float = None) -> dict:
        """Recognize Braille from base64 image with optional finger position."""
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                print("⚠️  Failed to decode image")
                return {'text': '', 'dotCount': 0, 'cellCount': 0, 'error': 'Failed to decode image'}
            
            h, w = image.shape[:2]
            print(f"📸 Image size: {w}x{h}")
            
            # Detect all dots
            dots = self.detect_braille_dots(image)
            print(f"🔍 Detected {len(dots)} dots")
            
            if len(dots) == 0:
                return {
                    'text': '',
                    'dotCount': 0,
                    'cellCount': 0,
                    'fingerCell': False,
                    'dots': [],
                    'message': 'No Braille dots detected - point camera at Braille text'
                }
            
            # Cluster into cells
            cells = self.cluster_dots_into_cells(dots)
            print(f"📦 Clustered into {len(cells)} cells")
            
            # If finger position provided, find nearest cell
            if finger_x is not None and finger_y is not None:
                finger_px = int(finger_x * w)
                finger_py = int(finger_y * h)
                print(f"👆 Finger at: ({finger_px}, {finger_py})")
                
                # Find cell nearest to finger
                min_dist = float('inf')
                nearest_cell = None
                
                for cell in cells:
                    # Cell center
                    cell_cx = sum(d[0] for d in cell) / len(cell)
                    cell_cy = sum(d[1] for d in cell) / len(cell)
                    
                    dist = ((cell_cx - finger_px)**2 + (cell_cy - finger_py)**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        nearest_cell = cell
                
                # Recognize character at finger position
                if nearest_cell and min_dist < 100:  # Within 100 pixels
                    char = self.dots_to_character(nearest_cell)
                    print(f"✓ Character at finger: '{char}' (distance: {min_dist:.1f}px)")
                    return {
                        'text': char,
                        'dotCount': len(dots),
                        'cellCount': len(cells),
                        'fingerCell': True,
                        'dots': dots[:10]  # Send first 10 dot positions for debugging
                    }
                else:
                    print(f"⚠️  No cell near finger (min distance: {min_dist:.1f}px)")
            
            # Otherwise, recognize all cells left to right
            recognized_text = ''
            for cell in cells:
                char = self.dots_to_character(cell)
                recognized_text += char
            
            print(f"✓ Full text: '{recognized_text}'")
            
            return {
                'text': recognized_text,
                'dotCount': len(dots),
                'cellCount': len(cells),
                'fingerCell': False,
                'dots': dots[:10]
            }
        except Exception as e:
            print(f"❌ Error in recognize_braille: {e}")
            import traceback
            traceback.print_exc()
            return {
                'text': '',
                'dotCount': 0,
                'cellCount': 0,
                'error': str(e)
            }
    
    async def handle_client(self, websocket):
        """Handle WebSocket client connection."""
        PORT = int(os.getenv('PORT', '8765'))
        print(f"✓ Server listening on port {PORT}")
        print(f"✓ Client connected")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data['type'] == 'predict_finger':
                        # Finger reading prediction
                        if 'sensor' not in data:
                            raise ValueError("No sensor data")
                        
                        sensor_data = data['sensor']
                        
                        if 'finger' not in sensor_data or 'imu' not in sensor_data:
                            raise ValueError("Missing finger or IMU data")
                        
                        if len(sensor_data['finger']) == 0 or len(sensor_data['imu']) == 0:
                            raise ValueError("Empty sensor data")
                        
                        print(f"📊 Finger prediction: {len(sensor_data['finger'])} samples")
                        
                        predictions = self.predict_finger(sensor_data, top_k=5)
                        
                        print(f"✓ Top prediction: {predictions[0]['text'][:50]}...")
                        
                        response = {
                            'type': 'prediction',
                            'mode': 'finger',
                            'predictions': predictions
                        }
                        await websocket.send(json.dumps(response))
                    
                    elif data['type'] == 'recognize_braille':
                        # Braille recognition
                        if 'image' not in data:
                            raise ValueError("No image data")
                        
                        # Get optional finger position
                        finger_x = data.get('finger_x', None)
                        finger_y = data.get('finger_y', None)
                        
                        print(f"🔤 Braille recognition request (finger: {finger_x is not None})")
                        
                        result = self.recognize_braille(data['image'], finger_x, finger_y)
                        
                        print(f"✓ Detected {result['dotCount']} dots, {result.get('cellCount', 0)} cells")
                        if result.get('text'):
                            print(f"   Recognized: '{result['text']}'")
                        
                        response = {
                            'type': 'braille_result',
                            'mode': 'braille',
                            'result': result
                        }
                        await websocket.send(json.dumps(response))
                    
                    elif data['type'] == 'ping':
                        await websocket.send(json.dumps({'type': 'pong'}))
                
                except ValueError as e:
                    print(f"⚠️  Validation error: {e}")
                    error_response = {
                        'type': 'error',
                        'message': str(e)
                    }
                    await websocket.send(json.dumps(error_response))
                
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    error_response = {
                        'type': 'error',
                        'message': str(e)
                    }
                    await websocket.send(json.dumps(error_response))
        
        except websockets.exceptions.ConnectionClosed:
            print(f"✗ Client disconnected")
        except Exception as e:
            print(f"❌ Connection error: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self, host: str = 'localhost', port: int = 8765):
        """Start WebSocket server."""
        print(f"\n🌐 Starting WebSocket server on ws://{host}:{port}")
        print(f"Supports: Finger Reading + Braille Recognition")
        print(f"Press Ctrl+C to stop\n")
        
        async with websockets.serve(self.handle_client, host, port):
            await asyncio.Future()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified inference server')
    parser.add_argument('--model', default='models/best_model.pt',
                       help='Path to finger reading model')
    parser.add_argument('--index', default='models/inference_index.pkl',
                       help='Path to embedding index')
    parser.add_argument('--host', default='localhost',
                       help='Server host')
    parser.add_argument('--port', type=int, default=8765,
                       help='Server port')
    parser.add_argument('--device', default=None,
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    server = UnifiedInferenceServer(
        model_path=args.model,
        index_path=args.index,
        device=args.device
    )
    
    try:
        asyncio.run(server.start(host=args.host, port=args.port))
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")


if __name__ == "__main__":
    main()