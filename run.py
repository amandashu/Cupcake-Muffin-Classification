import sys
import json

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/utils')

from remove import remove_ouputs
from scrape import scrape_data
from clean import clean_data

def main(targets):
    if 'clean' in targets:
        remove_ouputs()
        return

    if 'data' in targets:
        with open('config/chromedriver.json') as fh:
                    chromedriver_path = json.load(fh)['chromedriver_path']

        scrape_data(chromedriver_path)
        clean_data()
        return


    if 'data-scrape' in targets:
        with open('config/chromedriver.json') as fh:
                    chromedriver_path = json.load(fh)['chromedriver_path']

        scrape_data(chromedriver_path)

    if 'data-clean' in targets:
        clean_data()

    return

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
