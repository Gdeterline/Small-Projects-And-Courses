### Guide: Web Scraping with Python (BeautifulSoup & Requests)

This guide defines the basics of web scraping with Python using BeautifulSoup and Requests. It includes a mini-project related to NBA teams or players' stats after completing the exercises.

---

#### **Module 1: Introduction to Web Scraping with Python**

##### **1.1 What is Web Scraping?**
Web scraping refers to extracting data from websites. In Python, the popular libraries for web scraping are:

- **Requests**: Allows you to send HTTP requests to a website.
- **BeautifulSoup**: Parses the HTML content of the page to extract useful data.

##### **1.2 Required Libraries**
- **Install Requests and BeautifulSoup**:
  ```bash
  pip install requests beautifulsoup4
  ```

##### **1.3 Basic Web Scraping Workflow**
1. Send an HTTP request to a URL using `requests`.
2. Parse the content of the page with `BeautifulSoup`.
3. Extract specific data from the parsed HTML.

---

#### **Module 2: Understanding the Structure of a Web Page**

Before scraping, it's essential to understand how web pages are structured. Websites are usually built using **HTML**, which consists of:

- **Tags** (e.g., `<div>`, `<span>`, `<a>`).
- **Attributes** (e.g., `class`, `id`).
- **Content** (e.g., text inside a tag).

##### **2.1 Inspecting a Page Using Developer Tools**
To inspect the structure of a webpage, use browser tools:
- Right-click on a page element and choose "Inspect" (in Chrome or Firefox) to view the HTML structure.

For example, on the Giannis Antetokounmpo Wikipedia page, inspect the section with player information like his career stats.

---

#### **Module 3: Making The First Scrape**

##### **3.1 Scraping Giannis Antetokounmpo’s Wikipedia Page**
Let’s start with basic scraping from the Giannis Wikipedia page.

1. **Send an HTTP Request**:
   ```python
   import requests
   url = "https://en.wikipedia.org/wiki/Giannis_Antetokounmpo"
   response = requests.get(url)
   print(response.status_code)  # Should return 200 for a successful request
   ```

2. **Parse the HTML Content**:
   ```python
   from bs4 import BeautifulSoup
   soup = BeautifulSoup(response.content, 'html.parser')
   ```

3. **Find and Extract Data**:
   Let’s extract the title of the page.
   ```python
   title = soup.find('h1').text
   print(title)
   ```

---

#### **Module 4: Extracting Specific Data from a Page**

##### **4.1 Extracting a Player’s Stats**

On the Giannis page, there are career statistics in a table. We can extract them by targeting the `table` tag and then looking for rows (`<tr>`).

```python
# Find the stats table
stats_table = soup.find('table', {'class': 'wikitable'})

# Extract the rows
rows = stats_table.find_all('tr')
for row in rows:
    columns = row.find_all('td')
    if columns:
        stats = [column.text.strip() for column in columns]
        print(stats)
```

This will print out the stats of the player (Giannis Antetokounmpo in this case).

##### **4.2 Extracting Links**
It is possible to extract all the links from the page.
```python
links = soup.find_all('a', href=True)
for link in links:
    print(link['href'])
```

---

#### **Module 5: Working with Dynamic Content (Optional)**

Some websites load data dynamically (e.g., through JavaScript). To scrape such content, you may need to use **Selenium** or **Playwright**. However, for now, we’ll focus on static pages.

---

#### **Module 6: Mini Project - NBA Stats Web Scraping**

Now, we move on to the mini-project: **Scraping NBA Players/Teams Stats**. The idea is to apply everything learned and expand it to scrape data from various player pages.

##### **6.1 Define the Scope**
1. Player Name
2. Career Stats (from tables like games played, points, assists, etc.)
3. Links to external pages for further analysis.

##### **6.2 Setup The Project Directory**
Create a folder for the project:
```bash
mkdir nba_stats_scraper
cd nba_stats_scraper
```

##### **6.3 Write the Scraping Code**
Create a Python script to scrape the data from NBA players’ Wikipedia pages. For this, choose a list of players, for example:
- Giannis Antetokounmpo: `https://en.wikipedia.org/wiki/Giannis_Antetokounmpo`
- LeBron James: `https://en.wikipedia.org/wiki/LeBron_James`

##### **6.4 Example Code for Multiple Players**
```python
import requests
from bs4 import BeautifulSoup

# List of NBA players' URLs
players_urls = [
    "https://en.wikipedia.org/wiki/Giannis_Antetokounmpo",
    "https://en.wikipedia.org/wiki/LeBron_James"
]

def scrape_player_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract player name
    name = soup.find('h1').text.strip()
    
    # Extract career stats (example: first stats table)
    stats_table = soup.find('table', {'class': 'wikitable'})
    stats = []
    if stats_table:
        rows = stats_table.find_all('tr')
        for row in rows:
            columns = row.find_all('td')
            if columns:
                stats.append([column.text.strip() for column in columns])
    
    # Print the player's name and stats
    print(f"Stats for {name}:")
    for stat in stats:
        print(stat)

# Scrape data for all players
for player_url in players_urls:
    scrape_player_data(player_url)
```

##### **6.5 Saving the Data**
Store the scraped data in a CSV file.
```python
import csv

def save_to_csv(data, filename='nba_stats.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Example usage: saving player stats
save_to_csv(stats)
```

##### **6.6 Bonus: Scraping NBA Team Stats**
Instead of individual players, scrape stats from NBA team pages in a similar way. Check out pages like:
- `https://en.wikipedia.org/wiki/List_of_NBA_teams`

---

#### **Module 7: Conclusion & Next Steps**

1. **Improving Scraper**: Add more complex features, like pagination or handling JavaScript.
2. **Data Cleaning**: Use Pandas to clean and structure the scraped data for analysis.
3. **Building a Web Scraping Pipeline**: Automate the scraping process and handle errors gracefully (e.g., retries, timeouts).

---

### Exercise Recap

1. **Basic Scraping**: Scrape the title and the first stats table from Giannis’s Wikipedia page.
2. **Advanced Scraping**: Extract player stats, including career games, points, and assists.
3. **Mini Project**: Scrape NBA stats for multiple players, clean the data, and store it in a CSV.