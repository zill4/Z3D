import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    # Simple cube OBJ file
    with open(os.path.join(args.output, "cube.obj"), "w") as f:
        f.write("""v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v -0.5 0.5 0.5
v 0.5 0.5 0.5
f 1 2 3 4""")

if __name__ == "__main__":
    main()
