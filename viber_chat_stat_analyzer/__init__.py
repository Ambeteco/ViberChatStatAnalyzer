import argparse
import os
import re
from collections import Counter
from datetime import datetime

import emoji
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud


class ViberChatAnalyzer:
    def __init__(self, file_path):
        """Initialize the analyzer with the path to the Viber chat export file."""
        self.file_path = file_path
        self.df = None
        self.parse_chat_file()

    def parse_chat_file(self):
        """Parse the Viber chat export file into a pandas DataFrame."""
        messages = []

        with open(self.file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                pattern = (
                    r"^(\d{2}/\d{2}/\d{4}),(\d{2}:\d{2}:\d{2}),([^,]+),([^,]+),(.*)$"
                )
                match = re.match(pattern, line)

                if match:
                    date, time, sender, phone, message = match.groups()

                    date_obj = datetime.strptime(date + " " + time, "%d/%m/%Y %H:%M:%S")

                    messages.append(
                        {
                            "date": date_obj,
                            "sender": sender,
                            "phone": phone,
                            "message": message,
                            "year": date_obj.year,
                            "month": date_obj.month,
                            "day": date_obj.day,
                            "hour": date_obj.hour,
                            "day_of_week": date_obj.weekday(),
                            "is_media": message
                            in ["Фотоповідомлення", "Стікер", "Голосове повідомлення"],
                        }
                    )

        self.df = pd.DataFrame(messages)
        print(f"Parsed {len(self.df)} messages")

    def get_basic_stats(self):
        """Get basic statistics about the chat."""
        if self.df is None or len(self.df) == 0:
            return "No data available"

        total_messages = len(self.df)
        total_senders = self.df["sender"].nunique()
        date_range = (self.df["date"].min(), self.df["date"].max())
        duration_days = (date_range[1] - date_range[0]).days + 1

        media_count = self.df["is_media"].sum()
        text_count = total_messages - media_count

        sender_counts = self.df["sender"].value_counts()

        stats = {
            "total_messages": total_messages,
            "total_senders": total_senders,
            "date_range": date_range,
            "duration_days": duration_days,
            "messages_per_day": total_messages / duration_days,
            "text_messages": text_count,
            "media_messages": media_count,
            "sender_message_counts": sender_counts.to_dict(),
        }

        return stats

    def plot_activity_over_time(self, save_path=None):
        """Plot chat activity over time."""
        if self.df is None or len(self.df) == 0:
            return "No data available"

        daily_counts = self.df.set_index("date").resample("D").size()

        plt.figure(figsize=(12, 6))
        daily_counts.plot()
        plt.title("Chat Activity Over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Messages")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            return f"Saved to {save_path}"
        else:
            plt.show()

        return "Chart generated"

    def plot_activity_by_hour(self, save_path=None):
        """Plot chat activity by hour of day."""
        if self.df is None or len(self.df) == 0:
            return "No data available"

        hour_counts = self.df["hour"].value_counts().sort_index()

        plt.figure(figsize=(12, 6))
        hour_counts.plot(kind="bar")
        plt.title("Chat Activity by Hour of Day")
        plt.xlabel("Hour")
        plt.ylabel("Number of Messages")
        plt.xticks(range(24), [f"{h:02d}:00" for h in range(24)], rotation=45)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            return f"Saved to {save_path}"
        else:
            plt.show()

        return "Chart generated"

    def plot_activity_by_day_of_week(self, save_path=None):
        """Plot chat activity by day of week."""
        if self.df is None or len(self.df) == 0:
            return "No data available"

        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        day_counts = self.df["day_of_week"].value_counts().sort_index()
        day_counts.index = [day_names[i] for i in day_counts.index]

        plt.figure(figsize=(10, 6))
        day_counts.plot(kind="bar")
        plt.title("Chat Activity by Day of Week")
        plt.xlabel("Day")
        plt.ylabel("Number of Messages")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            return f"Saved to {save_path}"
        else:
            plt.show()

        return "Chart generated"

    def plot_sender_distribution(self, save_path=None):
        """Plot distribution of messages by sender."""
        if self.df is None or len(self.df) == 0:
            return "No data available"

        sender_counts = self.df["sender"].value_counts()

        plt.figure(figsize=(10, 6))
        sender_counts.plot(kind="bar")
        plt.title("Messages by Sender")
        plt.xlabel("Sender")
        plt.ylabel("Number of Messages")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            return f"Saved to {save_path}"
        else:
            plt.show()

        return "Chart generated"

    def plot_media_vs_text(self, save_path=None):
        """Plot distribution of media vs text messages."""
        if self.df is None or len(self.df) == 0:
            return "No data available"

        media_count = self.df["is_media"].sum()
        text_count = len(self.df) - media_count

        plt.figure(figsize=(8, 8))
        plt.pie(
            [text_count, media_count],
            labels=["Text", "Media"],
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.axis("equal")
        plt.title("Text vs Media Messages")

        if save_path:
            plt.savefig(save_path)
            return f"Saved to {save_path}"
        else:
            plt.show()

        return "Chart generated"

    def analyze_response_times(self):
        """Analyze response times between messages."""
        if self.df is None or len(self.df) == 0:
            return "No data available"

        df_sorted = self.df.sort_values("date")

        df_sorted["next_date"] = df_sorted["date"].shift(-1)
        df_sorted["next_sender"] = df_sorted["sender"].shift(-1)

        df_sorted["response_time"] = None
        mask = df_sorted["sender"] != df_sorted["next_sender"]
        df_sorted.loc[mask, "response_time"] = (
            df_sorted.loc[mask, "next_date"] - df_sorted.loc[mask, "date"]
        ).dt.total_seconds() / 60

        df_responses = df_sorted[df_sorted["response_time"].between(0, 24 * 60)]

        response_stats = df_responses.groupby("next_sender")["response_time"].agg(
            ["mean", "median", "min", "max", "count"]
        )

        return response_stats

    def plot_response_times(self, save_path=None):
        """Plot average response times for each sender."""
        response_stats = self.analyze_response_times()
        if isinstance(response_stats, str):
            return response_stats

        plt.figure(figsize=(10, 6))
        response_stats["mean"].plot(kind="bar")
        plt.title("Average Response Time by Sender")
        plt.xlabel("Sender")
        plt.ylabel("Average Response Time (minutes)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            return f"Saved to {save_path}"
        else:
            plt.show()

        return "Chart generated"

    def analyze_conversation_length(self):
        """Analyze conversation lengths (consecutive messages without long breaks)."""
        if self.df is None or len(self.df) == 0:
            return "No data available"

        df_sorted = self.df.sort_values("date")

        df_sorted["time_diff"] = df_sorted["date"].diff().dt.total_seconds() / 60

        conversation_threshold = 60
        df_sorted["new_conversation"] = df_sorted["time_diff"] > conversation_threshold
        df_sorted["conversation_id"] = df_sorted["new_conversation"].cumsum()

        conversation_stats = df_sorted.groupby("conversation_id").agg(
            start_time=("date", "min"),
            end_time=("date", "max"),
            duration=("date", lambda x: (x.max() - x.min()).total_seconds() / 60),
            messages=("date", "count"),
            participants=("sender", "nunique"),
        )

        return conversation_stats

    def generate_word_cloud(self, save_path=None):
        """Generate a word cloud from the chat messages, excluding the top 25 most frequent words."""
        if self.df is None or len(self.df) == 0:
            return "No data available"

        text_df = self.df[~self.df["is_media"]]

        all_text = " ".join(text_df["message"].fillna(""))

        tokens = word_tokenize(all_text.lower())
        word_counts = Counter(tokens)

        top25 = set([word for word, _ in word_counts.most_common(47)])

        default_sw = set(WordCloud().stopwords)
        combined_sw = default_sw.union(top25)

        wordcloud = WordCloud(
            width=1920,
            height=720,
            background_color="white",
            contour_width=1,
            contour_color="steelblue",
            collocations=False,
            max_words=500,
            stopwords=combined_sw,
        ).generate(all_text)

        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud from Chat Messages (top 25 words excluded)")

        if save_path:
            plt.savefig(save_path)
            return f"Saved to {save_path}"
        else:
            plt.show()

        return "Word cloud generated"

    def analyze_language_stats(self):
        """Analyze language statistics like common words, emojis, etc."""
        if self.df is None or len(self.df) == 0:
            return "No data available"

        text_df = self.df[~self.df["is_media"]]

        all_words = []
        for msg in text_df["message"].fillna(""):
            words = word_tokenize(msg.lower())
            all_words.extend(words)

        word_counts = Counter(all_words)

        all_emojis = []
        for msg in text_df["message"].fillna(""):
            emojis = [c for c in msg if c in emoji.EMOJI_DATA]
            all_emojis.extend(emojis)

        emoji_counts = Counter(all_emojis)

        stats = {
            "total_words": len(all_words),
            "unique_words": len(word_counts),
            "top_words": dict(word_counts.most_common(100)),
            "total_emojis": len(all_emojis),
            "unique_emojis": len(emoji_counts),
            "top_emojis": dict(emoji_counts.most_common(10)),
        }

        return stats

    def generate_heatmap_activity(self, save_path=None):
        """Generate a heatmap of activity by day of week and hour."""
        if self.df is None or len(self.df) == 0:
            return "No data available"

        if "hour" not in self.df.columns:
            self.df["hour"] = self.df["date"].dt.hour
        if "day_of_week" not in self.df.columns:
            self.df["day_of_week"] = self.df["date"].dt.dayofweek

        activity = pd.crosstab(self.df["day_of_week"], self.df["hour"])

        plt.figure(figsize=(15, 8))
        sns.heatmap(activity, cmap="YlGnBu", annot=True, fmt="d")

        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        plt.yticks(np.arange(0.5, len(day_names)), day_names)
        plt.xticks(np.arange(0.5, 24), [f"{h:02d}:00" for h in range(24)], rotation=45)

        plt.title("Chat Activity Heatmap by Day and Hour")
        plt.ylabel("Day of Week")
        plt.xlabel("Hour of Day")

        if save_path:
            plt.savefig(save_path)
            return f"Saved to {save_path}"
        else:
            plt.show()

        return "Heatmap generated"

    def generate_all_insights(self, output_dir=None):
        """Generate all possible insights and save them to files."""
        if output_dir is None:
            output_dir = "."

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        stats = self.get_basic_stats()
        with open(f"{output_dir}/basic_stats.txt", "w", encoding="utf-8") as f:
            f.write("Basic Chat Statistics\n")
            f.write("====================\n\n")
            for key, value in stats.items():
                if key != "sender_message_counts":
                    f.write(f"{key}: {value}\n")
            f.write("\nMessages by sender:\n")
            for sender, count in stats["sender_message_counts"].items():
                f.write(f"  {sender}: {count}\n")

        self.plot_activity_over_time(f"{output_dir}/activity_over_time.png")

        self.plot_activity_by_hour(f"{output_dir}/activity_by_hour.png")

        self.plot_activity_by_day_of_week(f"{output_dir}/activity_by_day.png")

        self.plot_sender_distribution(f"{output_dir}/sender_distribution.png")

        self.plot_media_vs_text(f"{output_dir}/media_vs_text.png")

        response_stats = self.analyze_response_times()
        if not isinstance(response_stats, str):
            response_stats.to_csv(f"{output_dir}/response_times.csv")
            self.plot_response_times(f"{output_dir}/response_times.png")

        conversation_stats = self.analyze_conversation_length()
        if not isinstance(conversation_stats, str):
            conversation_stats.to_csv(f"{output_dir}/conversation_stats.csv")

        self.generate_word_cloud(f"{output_dir}/word_cloud.png")

        language_stats = self.analyze_language_stats()
        if not isinstance(language_stats, str):
            with open(f"{output_dir}/language_stats.txt", "w", encoding="utf-8") as f:
                f.write("Language Statistics\n")
                f.write("==================\n\n")
                for key, value in language_stats.items():
                    if key not in ["top_words", "top_emojis"]:
                        f.write(f"{key}: {value}\n")
                f.write("\nTop words:\n")
                for word, count in language_stats["top_words"].items():
                    f.write(f"  {word}: {count}\n")
                f.write("\nTop emojis:\n")
                for emoji_, count in language_stats["top_emojis"].items():
                    f.write(f"  {emoji_}: {count}\n")

        self.generate_heatmap_activity(f"{output_dir}/activity_heatmap.png")

        return f"All insights generated and saved to {output_dir}"


def main():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    parser = argparse.ArgumentParser(
        description="Analyze a Viber chat export and generate reports."
    )
    parser.add_argument("input", help="Path to Viber chat export text file")
    parser.add_argument(
        "-o", "--output", default="viber_analysis_results", help="Output directory"
    )
    args = parser.parse_args()

    analyzer = ViberChatAnalyzer(args.input)
    result = analyzer.generate_all_insights(args.output)
    print(result)


if __name__ == "__main__":
    main()
