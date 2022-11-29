import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int)
args = parser.parse_args()

def ff(d):
    print(d)

if __name__ == "__main__":
    ff(args.dim)