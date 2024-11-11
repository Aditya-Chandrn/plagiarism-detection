def extract_title_and_abstract(md_file):
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.readlines()

    # title = None
    abstract = None

    # Assuming the title is the first line
    # if content:
    #     title = content[0].strip()  # Get the first line as the title

    # Loop through the content to find the abstract
    for i in range(len(content)):
        if "abstract" in content[i].lower():  # Find line containing "abstract"
            # Assuming the abstract follows immediately
            abstract_lines = []
            # Collect lines for the abstract until a new section starts
            for j in range(i + 1, len(content)):
                if content[j].strip() == "" or content[j].startswith("#"):  # Stop at empty line or new header
                    break
                abstract_lines.append(content[j].strip())   
            abstract = " ".join(abstract_lines)
            break

    return abstract

# Example usage
md_file = "./new.md"
abstract = extract_title_and_abstract(md_file)

# print("Title:", title)
# print("Abstract:", abstract)
# ======================================================================================================================
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_words = [word for word in word_tokens if word.isalnum() and word not in stop_words]
    return [word for word, _ in Counter(filtered_words).most_common(10)]  # Return just the words

# def setup_driver():
#     options = webdriver.ChromeOptions()
#     # options.add_argument('--headless')  # Uncomment to run headless
#     driver = webdriver.Chrome(options=options)
#     return driver

# def search_and_scrape(keywords, num_pages=5):
#     driver = setup_driver()
#     all_results = []
    
#     try:
#         # Ensure keywords is a string
#         if isinstance(keywords, (list, tuple)):
#             keywords = ' '.join(keywords)  # Join if it's a list or tuple
        
#         print(f"Searching for: {keywords}")  # Debugging line to check keywords

#         # Navigate to Google
#         driver.get('https://www.google.com')

#         # Find the search box, enter keywords, and submit
#         search_box = driver.find_element(By.NAME, 'q')
#         search_box.send_keys(keywords)  # Send keywords
#         search_box.send_keys(Keys.RETURN)  # Submit the search

#         for _ in range(num_pages):
#             time.sleep(2)  # Wait for results to load
#             results = driver.find_elements(By.XPATH, '//div[@class="g"]')
            
#             for result in results:
#                 try:
#                     title = result.find_element(By.TAG_NAME, 'h3').text  # Extract title
#                     link = result.find_element(By.TAG_NAME, 'a').get_attribute('href')  # Extract link
#                     all_results.append((title, link))
#                 except Exception as e:
#                     print(f"Error extracting result: {e}")

#             # Find and click the "Next" button to go to the next page of results
#             try:
#                 next_button = driver.find_element(By.XPATH, '//a[@id="pnnext"]')
#                 next_button.click()
#             except Exception as e:
#                 print("No more pages or error finding next button:", e)
#                 break  # Exit loop if there are no more pages

#     finally:
#         driver.quit()  # Close the browser

#     print(all_results)

#     return all_results

# # Example usage
# print(abstract)
keywords = extract_keywords(abstract)
# print(keywords)
# results = search_and_scrape(keywords)

# # Print all results
# for title, link in results:
#     print(f'Title: {title}\nLink: {link}\n')


if isinstance(keywords, (list, tuple)):
            keywords = ' '.join(keywords)  # Join if it's a list or tuple


import requests

API_KEY = 'AIzaSyDia0iALgUnDwLev4TAwUlzkKc7SPrATTA'  # Replace with your actual API Key
CSE_ID = '64ce8b3cf1cc14e49'  # Replace with your Custom Search Engine ID
query = keywords  # Keywords from your abstract
num_results = 100  # Number of results you want (max 100)

results = []


for start in range(1, num_results, 10):  # 1, 11, 21, ... up to num_results
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CSE_ID}&start={start}"
    response = requests.get(url).json()
    
    if 'items' in response:
        results.extend(response['items'])

# Process the results
i=1
for result in results:
    # print(result['title'])
    print(f"\n{i}. Title: {result['title']}")
    print(f"\nLink: {result['link']}")
    print(f"\nSnippet: {result['snippet']}\n")
    i+=1
    # print(result['snippet'])

# with open("urls.txt", "w", encoding='utf-8') as f:
#     for url in results:
#         f.write(f"{url['link']}\n")