"""
Quick Token Analysis Runner
Simple script to analyze your epoch 20 results with detailed token breakdowns
"""

import os
import sys

def main():
    """Run enhanced token analysis on your perfect results"""
    
    print("🚀 Starting Enhanced Token Analysis...")
    print("="*60)
    print("📝 This will analyze your epoch 20 transliterations with:")
    print("   ✅ Character-by-character token breakdown")
    print("   ✅ Token IDs and vocabulary mappings")
    print("   ✅ Attention weight analysis")
    print("   ✅ Character alignment visualization")
    print("   ✅ Confidence scores and accuracy metrics")
    print("   ✅ Comprehensive evaluation report")
    print()
    
    try:
        # Import and run the enhanced evaluation
        from enhanced_evaluate import main as run_evaluation
        
        # Run the analysis
        run_evaluation()
        
        print("\\n🎉 Analysis completed successfully!")
        print("\\n📄 Check the following files for results:")
        print("   📊 logs/enhanced_analysis/evaluation_statistics.json")
        print("   📝 logs/enhanced_analysis/detailed_token_analysis.json")
        print("   📈 logs/enhanced_analysis/evaluation_report.md")
        print("   🎨 logs/enhanced_analysis/attention_heatmap_sample1.png")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("   pip install torch matplotlib seaborn numpy pandas")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure the model checkpoint and config files exist:")
        print("   - models/best_model.pth")
        print("   - improved_config.json")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()