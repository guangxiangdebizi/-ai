import wikipedia
import os

# Set the language of Wikipedia articles
wikipedia.set_lang("en")


def save_wiki_page(title):
    try:
        # Fetch page content
        page = wikipedia.page(title)
        content = page.content

        # Define file path
        file_path = os.path.join(os.getcwd(), f"{title}.txt")

        # Write content to a text file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Page saved as '{file_path}'")
    except wikipedia.exceptions.PageError:
        print("Page not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
save_wiki_page("United_Kingdom")
