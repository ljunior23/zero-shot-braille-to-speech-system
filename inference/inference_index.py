import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from training.model import ContrastiveModel
from inference.nearest_neighbor import EmbeddingIndex


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build 128D inference index')
    parser.add_argument('--model', default='models/best_model.pt',
                       help='Trained model path')
    parser.add_argument('--audio-embeddings', default='data/features/audio_embeddings.npy',
                       help='768D audio embeddings')
    parser.add_argument('--texts', default='data/features/audio_texts.txt',
                       help='Text file')
    parser.add_argument('--output', default='models/inference_index.pkl',
                       help='Output index path')
    parser.add_argument('--device', default=None,
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device')
    
    args = parser.parse_args()
    
    # Device
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print("="*60)
    print("Building 128D Inference Index")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Audio embeddings: {args.audio_embeddings}")
    print()
    
    # Load model
    print("Loading model...")
    model = ContrastiveModel(
        sensor_input_dim=5,
        audio_input_dim=768,
        embedding_dim=128
    )
    
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Load audio embeddings (768D)
    print("\nLoading audio embeddings...")
    audio_embeddings_768 = np.load(args.audio_embeddings)
    print(f"✓ Loaded {len(audio_embeddings_768)} embeddings (768D)")
    
    # Load texts
    with open(args.texts, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f]
    print(f"✓ Loaded {len(texts)} texts")
    
    # Project to 128D using audio projector
    print("\nProjecting to 128D...")
    audio_embeddings_128 = []
    
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(audio_embeddings_768), batch_size):
            batch = audio_embeddings_768[i:i+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            
            # Project using audio projector
            projected = model.audio_projector(batch_tensor)
            audio_embeddings_128.append(projected.cpu().numpy())
            
            if (i + batch_size) % 1000 == 0:
                print(f"  Processed {min(i + batch_size, len(audio_embeddings_768))}/{len(audio_embeddings_768)}...")
    
    audio_embeddings_128 = np.vstack(audio_embeddings_128)
    print(f"✓ Projected to 128D: {audio_embeddings_128.shape}")
    
    # Build index
    print("\nBuilding FAISS index...")
    index = EmbeddingIndex(dimension=128)
    index.add_embeddings(audio_embeddings_128, texts)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(str(output_path))
    
    print(f"\n✓ Saved inference index to {output_path}")
    print("\nNow use this index with:")
    print(f"  python inference/inference_server.py --index {output_path}")


if __name__ == "__main__":
    main()