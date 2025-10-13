import requests
import logging
import time


# WIKIDATA QUERY UTILS

def run_wd_query(query: str):
    """
    This function queries the Wikidata-API(https://www.wikidata.org/wiki/Wikidata:Data_access#Wikidata_Query_Service)
    """
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/json"}
    data = ""

    # In case of rate limit error, try again after 2 seconds
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, params={"query": query})
            data = response.json()
            return data["results"]["bindings"]

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}\ndata: {data}")
            if attempt == 2:
                logging.error(f"Querying failed! {e}")
                return None
            time.sleep(3)


def find_wd_id(word_combo: str):
    """
    Find the Q-ID of an entity in Wikidata by its name.
    :param word_combo: prospective entity
    :return: Q-ID of the entity or None if not found
    """
    query = f"""
    SELECT ?item WHERE {{
      ?item rdfs:label "{word_combo.lower()}"@en .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """
    print("Querying for entity:", word_combo)
    results = run_wd_query(query)
    if results:
        return results[0]["item"]["value"].split("/")[-1]
    else:
        return None


# CONCEPTNET UTILS

def query_conceptnet(word: str, lang="en") -> dict:
    """
    Query ConceptNet for a given word and language.
    :param word: word to query
    :param lang: language code (default: "en")
    :return: response from ConceptNet API
    """
    word = word.replace(' ', '_').lower()
    url = f"https://api.conceptnet.io/c/{lang}/{word}"
    try:
        response = requests.get(url).json()
    except requests.exceptions.JSONDecodeError:
        response = {}
        logging.error(f"Failed to decode JSON response from ConceptNet for word: {word}")
    return response


if __name__ == "__main__":
    # Example usage
    term = "dry weight"
    print(f"'{term}' in Wikidata:", find_wd_id(term))
    boy = "boy"
    print(f"'{boy}' in ConceptNet: ", query_conceptnet(boy))