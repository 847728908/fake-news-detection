import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import cycle

def draw_sentiment_visualizations():
    """
    Read sentiment analysis result txt file, generate and save a visualization chart with two subplots:
    1. Bar chart of sentiment distribution
    2. Boxplot of sentiment intensity distribution
    """
    # --- 1. Environment Setup ---
    # Set font to support Chinese display
    try:
        # Prefer 'Microsoft YaHei', fallback to 'SimHei'
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
        # Fix minus sign display issue
        plt.rcParams['axes.unicode_minus'] = False
        print("Chinese font set successfully.")
    except Exception:
        print("Warning: Specified Chinese fonts ('Microsoft YaHei', 'SimHei') not found, Chinese characters in the chart may not display properly.")

    # Define file paths
    file_path = r'C:\Users\84772\Desktop\try\result_sentiment2_combine_text_newprompt.txt'
    output_path = 'sentiment_visualization.png'

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    # --- 2. Data Loading and Preprocessing ---
    # Read data using tab as separator
    df = pd.read_csv(file_path, sep='\t')

    # Extract news type (fake or real) from file name
    df['type'] = df['file_name'].apply(lambda x: 'fake' if '_fake_' in x else 'real')
    
    # Clean possible leading/trailing spaces in sentiment labels
    df['sentiment'] = df['sentiment'].str.strip()

    # --- 3. Plotting (Beautified Version) ---
    # Create a 1x2 subplot layout and set the overall title
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Sentiment Analysis Visualization of Real and Fake News', fontsize=22, y=0.98)

    # --- Define Aesthetic Elements ---
    sentiment_order = ['positive', 'neutral', 'negative']
    # Use softer color palettes
    custom_palette_3 = sns.color_palette("Pastel1", 3)
    custom_palette_2 = sns.color_palette("Pastel1", 2)
    # Define hatches
    hatches = ['//', '..', 'xx']


    # --- Subplot 1: Sentiment Distribution (Bar Chart) ---
    ax1 = sns.countplot(x='type', hue='sentiment', data=df, ax=axes[0],
                        order=['real', 'fake'], hue_order=sentiment_order,
                        palette=custom_palette_3)

    # Manually add transparency and hatches to bars
    hatch_cycle = cycle(hatches)
    for bar in ax1.patches:
        bar.set_alpha(0.7)
        bar.set_hatch(next(hatch_cycle))

    ax1.set_title('Sentiment Distribution', fontsize=18, pad=20)
    ax1.set_xlabel('News Type', fontsize=14)
    ax1.set_ylabel('Count', fontsize=14)
    ax1.set_xticklabels(['Real News', 'Fake News'], fontsize=12)
    ax1.legend(title='Sentiment')

    # Add count labels on bars
    for p in ax1.patches:
        if p.get_height() > 0:
            ax1.annotate(f'{int(p.get_height())}',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center',
                         xytext=(0, 9),
                         textcoords='offset points',
                         fontsize=11,
                         color='dimgray')

    # --- Subplot 2: Sentiment Intensity Distribution (Boxplot) ---
    # Use the same palette for boxplot
    sns.boxplot(x='type', y='intensity', data=df, ax=axes[1],
                order=['real', 'fake'], palette=custom_palette_2)

    # Add transparency and hatches to boxes
    for i, artist in enumerate(axes[1].artists):
        artist.set_alpha(0.7)
        artist.set_hatch(hatches[i])

    # Overlay stripplot to show data distribution
    sns.stripplot(x='type', y='intensity', data=df, ax=axes[1],
                  order=['real', 'fake'], color="gray", size=3.5, alpha=0.5)

    axes[1].set_title('Sentiment Intensity Distribution', fontsize=18, pad=20)
    axes[1].set_xlabel('News Type', fontsize=14)
    axes[1].set_ylabel('Sentiment Intensity', fontsize=14)
    axes[1].set_xticklabels(['Real News', 'Fake News'], fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    # --- 4. Save and Output ---
    # Adjust layout to avoid title overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save chart to new file with higher resolution
    output_path = 'sentiment_visualization.png'
    plt.savefig(output_path, dpi=300)
    print(f"Beautified visualization chart successfully saved to: {output_path}")

if __name__ == '__main__':
    # Run main function
    draw_sentiment_visualizations()
