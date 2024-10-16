# Interactive scatter

Used to visualise UMAP and images connected to the scattered points.

# Installation

```
pip install -r requirements.txt
```

# Running

Currently only supported option is to run interactive scatter function in the python script.

Example:

```python
from interscat.interactive_scatter import interactive_scatter

path_csv = r"./umap.csv"
path_images = r"./images/"
interactive_scatter(path_csv, path_images)
```

alternatively the pd.Dataframe object can be passed instead of the path:

```python
import pandas as pd
from interscat.interactive_scatter import interactive_scatter

df = pd.read_csv(r"./umap.csv")
path_images = r"./images/"
interactive_scatter(df, path_images)
```

After running the code, in the execution window, information will be shown e.g. 
"Dash is running on http://127.0.0.1:8050/". Click on the given address to open the visualisation
in your default browser.