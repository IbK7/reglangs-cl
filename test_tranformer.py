import torch
from models.transformer_relative import Transformer

def test_transformer_modes():
    """Test both absolute and relative positional encoding modes"""
    
    print("=" * 80)
    print("TESTING TRANSFORMER IMPLEMENTATIONS")
    print("=" * 80)
    
    # Test parameters
    batch_size = 3
    seq_len = 12
    input_size = 16
    hidden_size = 32
    output_size = 8
    max_out_length = 4
    
    # Important: lengths[i] + max_out_length must be <= seq_len
    # This ensures there's room in the sequence for the output positions
    lengths = torch.tensor([4, 5, 3])  # Input ends at these positions
    out_lengths = torch.tensor([4, 3, 4])  # Actual output lengths
    
    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Output size: {output_size}")
    print(f"  Max output length: {max_out_length}")
    print(f"  Input lengths: {lengths.tolist()}")
    print(f"  Output lengths: {out_lengths.tolist()}")
    
    # Create random input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Test 1: Absolute Positional Encoding
    print("\n" + "=" * 80)
    print("TEST 1: Absolute Positional Encoding")
    print("=" * 80)
    
    model_abs = Transformer(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        use_relative_positions=False
    )
    
    try:
        with torch.no_grad():
            output_abs = model_abs(x, lengths, out_lengths, max_out_length)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {output_abs.shape}")
        print(f"  Expected:     torch.Size([{batch_size}, {max_out_length}, {output_size}])")
        
        assert output_abs.shape == (batch_size, max_out_length, output_size), \
            f"Output shape mismatch! Got {output_abs.shape}"
        print(f"✓ Output shape correct")
        
        assert not torch.isnan(output_abs).any(), "NaN values detected!"
        print(f"✓ No NaN values")
        
        print(f"\n✅ Absolute encoding test PASSED")
        
    except Exception as e:
        print(f"\n❌ Absolute encoding test FAILED")
        print(f"Error: {e}")
        raise
    
    # Test 2: Relative Positional Encoding
    print("\n" + "=" * 80)
    print("TEST 2: Relative Positional Encoding (Transformer-XL)")
    print("=" * 80)
    
    model_rel = Transformer(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        use_relative_positions=True
    )
    
    try:
        with torch.no_grad():
            output_rel = model_rel(x, lengths, out_lengths, max_out_length)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {output_rel.shape}")
        print(f"  Expected:     torch.Size([{batch_size}, {max_out_length}, {output_size}])")
        
        assert output_rel.shape == (batch_size, max_out_length, output_size), \
            f"Output shape mismatch! Got {output_rel.shape}"
        print(f"✓ Output shape correct")
        
        assert not torch.isnan(output_rel).any(), "NaN values detected!"
        print(f"✓ No NaN values")
        
        print(f"\n✅ Relative encoding test PASSED")
        
    except Exception as e:
        print(f"\n❌ Relative encoding test FAILED")
        print(f"Error: {e}")
        raise
    
    # Test 3: Gradient Flow
    print("\n" + "=" * 80)
    print("TEST 3: Gradient Flow")
    print("=" * 80)
    
    # Test with requires_grad
    x_grad = x.clone().requires_grad_(True)
    
    # Absolute
    output_abs = model_abs(x_grad, lengths, out_lengths, max_out_length)
    loss_abs = output_abs.sum()
    loss_abs.backward()
    
    assert x_grad.grad is not None, "Gradients not computed for absolute model!"
    assert not torch.isnan(x_grad.grad).any(), "NaN gradients in absolute model!"
    print(f"✓ Absolute model: gradients flow correctly")
    
    # Relative
    x_grad = x.clone().requires_grad_(True)
    output_rel = model_rel(x_grad, lengths, out_lengths, max_out_length)
    loss_rel = output_rel.sum()
    loss_rel.backward()
    
    assert x_grad.grad is not None, "Gradients not computed for relative model!"
    assert not torch.isnan(x_grad.grad).any(), "NaN gradients in relative model!"
    print(f"✓ Relative model: gradients flow correctly")
    
    print(f"\n✅ Gradient flow test PASSED")
    
    # Test 4: Parameter Count
    print("\n" + "=" * 80)
    print("TEST 4: Parameter Count Comparison")
    print("=" * 80)
    
    params_abs = sum(p.numel() for p in model_abs.parameters())
    params_rel = sum(p.numel() for p in model_rel.parameters())
    ratio = params_rel / params_abs
    
    print(f"  Absolute model: {params_abs:,} parameters")
    print(f"  Relative model: {params_rel:,} parameters")
    print(f"  Ratio: {ratio:.2f}x")
    print(f"  Increase: {params_rel - params_abs:,} parameters (+{(ratio-1)*100:.1f}%)")
    
    # Test 5: Different Sequence Lengths
    print("\n" + "=" * 80)
    print("TEST 5: Generalization to Different Lengths")
    print("=" * 80)
    
    for test_seq_len in [10, 20, 30]:
        if test_seq_len < max_out_length + max(lengths):
            continue
            
        x_test = torch.randn(batch_size, test_seq_len, input_size)
        
        try:
            with torch.no_grad():
                out_abs = model_abs(x_test, lengths, out_lengths, max_out_length)
                out_rel = model_rel(x_test, lengths, out_lengths, max_out_length)
            
            assert out_abs.shape == (batch_size, max_out_length, output_size)
            assert out_rel.shape == (batch_size, max_out_length, output_size)
            
            print(f"✓ Sequence length {test_seq_len}: Both models work")
            
        except Exception as e:
            print(f"✗ Sequence length {test_seq_len}: Error - {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✅")
    print("=" * 80)
    print("\nBoth transformer implementations are working correctly.")
    print("You can now train and compare them on your task!")
    print("\nUsage:")
    print("  # Absolute positions (original)")
    print("  model = Transformer(..., use_relative_positions=False)")
    print()
    print("  # Relative positions (Transformer-XL)")
    print("  model = Transformer(..., use_relative_positions=True)")
    print("=" * 80)


if __name__ == "__main__":
    test_transformer_modes()