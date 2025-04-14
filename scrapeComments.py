import praw
import pandas as pd

reddit = praw.Reddit(
    client_id="Taqch-lnFMuxAxXa-zE4Gg",
    client_secret="wTu8ocPvwg5MZ1kOP-cBxYMKihYZow",
    user_agent="YourAppName/0.1 by Leather_Kale_2468",
)

try:
    subreddit = reddit.subreddit("all")
    search_query = "women in tech"
    data = []
    for post in subreddit.search():
        data.append(
            {
                "Post Title": post.title,
                "Post Subreddit": post.subreddit,
                "Score": post.score,
                "Comments": post.num_comments,
            }
        )

except Exception as e:
    print(f"Error: {e}")

df = pd.DataFrame(data)
df.to_csv("women_in_tech_comments.csv", index=False)
print("Scraping complete! Data saved to women_in_tech_comments.csv")
