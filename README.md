## Cupcake Muffin Classification
Scraping, cleaning, and classifying cupcake and muffin recipes. This project is inspired by this [SVM tutorial](https://www.youtube.com/watch?v=N1vOgolbjSc&feature=youtu.be).

Cupcakes and muffins may seem similar, but by comparing the ingredients of muffins and cupcakes from many online recipes, the data suggests otherwise. See [here](https://github.com/amandashu/Cupcake-Muffin-Classification/blob/main/reports/analysis.ipynb) for analysis Jupyter notebook.

## Data
The data is scraped from baking blogs [Baking Bites](https://bakingbites.com/), [Sally's Baking Addition](https://sallysbakingaddiction.com/), and [The Baker Chick](https://www.thebakerchick.com/). See [here](https://github.com/amandashu/Cupcake-Muffin-Classification/blob/main/reports/clean.md) for details on data preparation.

## Code Organization
#### Reports
The `/reports` folder contains:
- `clean.md`: a detailed explanation of the data cleaning and preparation steps
- `analysis.ipynb`: a Jupyter Notebook containing data analysis and results for a few classification algorithms

#### Source
The `/src` folder contains subfolders `data`, `utils`, and `models`.

In the `/src/data` folder:
- `scrape.py`: the web-scraping script that writes data files into `/data` folder. It outputs pickle files in the format `<baking blog abbreviation>_<cupcake or muffin>.pickle`, which contain the individual links to each recipe. Also outputted is `recipes.csv`, which is a data file containing columns `link`,  `type` (cupcake or muffin), and `ingredients` (list of ingredients scraped).
- `clean.py`: contains functions that cleans data, as described in `reports/clean.md`. It outputs `recipes_clean.csv` to the data folder

In the `src/utils` folder:
- `remove.py`: contains function `remove_data` that implements the standard target `clean`

In the `src/models` folder:
- `knn.py`: contains implementation of the k nearest neighbor algorithm
- `mlp.py`: contains implementation of multilayer perception classifier. After training MLP, the best model parameters are saved into `/results/model.pt`.

## Run the Results
To get the cleaned data, run the below command. This will run the webscraping and cleaning, while the targets `data-scrape` and `data-clean` will do each respectively.
```console
python run.py data
```

Note: To run the webscraping, this assumes that there is a file `/config/chromedriver.json` that specifies where the path to the downloaded chromedriver.exe file for your Chrome version lies. It might look like this:
```console
{
  "chromedriver_path" : "C:/Users/JohnDoe/Downloads/chromedriver_win32/chromedriver.exe"
}
```

Standard target `clean` is also implemented, and it will delete the `/data` and `/results` folder.

After running the data gathering and preprocessing, `/reports/analysis.ipynb` can be run to replicate model results.
