import sys

def test_imports():
    try:
        print("Attempting to import 'qml'...")
        import qml
        print("Main 'qml' module imported successfully.")
        
        # Check submodules if needed
        # import qml.a
        # import qml.b
        
    except ImportError as e:
        print(f"Import Error: {e}")
        sys.exit(1)
    except AttributeError as e:
        print(f"Attribute Error (possible missing __all__): {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during import: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()
