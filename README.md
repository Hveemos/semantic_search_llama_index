create a virtual environment, e.g.

`python -m venv .venv -r requirements.txt`

Run through the notebook. The first time it will have to compute the embeddings so it might take a while.

he first run might be slow. 
On a MacBook Pro M1 you might expect:

- Cell 1: 1 minutes
- Cell 2: 2:30 minutes
- Cell 3: 7 minutes with mps (this might take hours if no hardware acceleration is available)