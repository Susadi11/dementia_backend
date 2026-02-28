"""
Install BERT dependencies for improved text parsing

This installs:
1. transformers - Hugging Face transformers library (BERT models)
2. torch - PyTorch (required by transformers)
3. dateparser - Advanced date/time parsing from natural language

After installation, the system will automatically use BERT for better accuracy!
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    print(f"\n📦 Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    print("=" * 80)
    print("🤖 BERT TEXT PARSER - DEPENDENCY INSTALLER")
    print("=" * 80)
    print("\nThis will install advanced NLP libraries for better reminder parsing.")
    print("\nPackages to install:")
    print("  1. transformers - BERT models for entity recognition")
    print("  2. torch - PyTorch (required by transformers)")
    print("  3. dateparser - Natural language date/time parsing")
    print("\nTotal size: ~2-3 GB (includes pre-trained models)")
    print("=" * 80)
    
    response = input("\n⚠️  Proceed with installation? (y/n): ")
    if response.lower() != 'y':
        print("❌ Installation cancelled.")
        return
    
    packages = [
        "torch",
        "transformers",
        "dateparser"
    ]
    
    results = {}
    for package in packages:
        results[package] = install_package(package)
    
    print("\n" + "=" * 80)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 80)
    
    for package, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status} - {package}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\n" + "=" * 80)
        print("✅ ALL PACKAGES INSTALLED SUCCESSFULLY!")
        print("=" * 80)
        print("\n🚀 BERT text parser is now active!")
        print("\nThe system will automatically use BERT for:")
        print("  • Better date/time extraction")
        print("  • Named entity recognition (medication names)")
        print("  • Context-aware parsing")
        print("  • Natural language understanding")
        print("\nNo code changes needed - it's automatic! 🎉")
        print("\n" + "=" * 80)
    else:
        print("\n" + "=" * 80)
        print("⚠️  SOME PACKAGES FAILED")
        print("=" * 80)
        print("\nThe system will continue to work using regex-based parsing.")
        print("You can try installing failed packages manually:")
        for package, success in results.items():
            if not success:
                print(f"  pip install {package}")
        print("=" * 80)

if __name__ == "__main__":
    main()
