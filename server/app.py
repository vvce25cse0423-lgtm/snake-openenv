import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import EnvHandler
from http.server import HTTPServer

def main():
    port = 7860
    print("Snake OpenEnv server running at http://localhost:" + str(port))
    HTTPServer(("0.0.0.0", port), EnvHandler).serve_forever()

if __name__ == "__main__":
    main()
