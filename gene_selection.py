import re
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from tqdm import tqdm
from tryptag import TrypTag, CellLine

"""
This script uses the TrypTag localisation search to find genes that are localised to the paraflagellar rod.
Im using paraflagellar rod since "there are no clear examples of proteins concentrated in 
the flagellar cytoplasm in trypanosomatids." (Cellular landmarks of Trypanosoma brucei and Leishmania mexicana))

It uses Selenium to scrape the percentile score for each gene from the TrypTag website.
Then it creates a ranking for the percentile scores.
Then I manually select the top genes and add them to the list of genes to be analysed, and to be part of the dataset.
I decide based on the image clarity. Only the paraflagellar rod/flagellum should be brightly highlighted and the rest of the cell should be dark.
"""


def find_genes():
    tryptag = TrypTag()
    genes = tryptag.localisation_search(query_term="paraflagellar rod")

    base_url = "http://preview.tryptag.org/?query="

    percentile_scores = {}
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)

    first = True

    for gene in tqdm(genes):
        name = gene.gene_id
        try:
            full_url = base_url + name
            driver.get(full_url)

            if first:
                # wait for the page to load
                time.sleep(1)

                # click the accept terms button
                accept_button = driver.find_element(By.XPATH, '/html/body/div[3]/div/div[2]/div/div[3]/p/a')
                accept_button.click()
                first = False

            # wait for the page to load
            time.sleep(1)

            # find the element using JavaScript descriptor
            element = driver.find_element(By.CSS_SELECTOR,
                                          f"div.image-container > div.material-content.material-card-imageOverlay > p > span:nth-child(5)")
            percentile = re.search(r'(\d+)', element.text)
            if percentile:
                percentile_scores[name] = [gene, int(percentile.group(1)), full_url]
        except:
            print(f"Error for {name}")
            pass

    driver.quit()

    sorted_percentile_scores = {k: v for k, v in
                                sorted(percentile_scores.items(), key=lambda item: item[1][1], reverse=True)}

    for i, (k, v) in enumerate(sorted_percentile_scores.items()):
        print(f"{i + 1}. {k}: {v}")