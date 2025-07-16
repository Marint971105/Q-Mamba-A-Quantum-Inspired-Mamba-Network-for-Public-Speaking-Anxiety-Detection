import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def load_single_modality_predictions(file_path):
    """Load individual modality prediction CSV"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path}: {df.shape[0]} samples")
        return df
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
        return None


def load_all_predictions(file_path):
    """Load the main CSV file containing all predictions"""
    df = pd.read_csv(file_path)
    print(f"Loaded main prediction file: {df.shape}")
    return df


def merge_prediction_data(main_df, text_df, audio_df, video_df):
    """Merge prediction data from all modality files"""
    result_df = main_df.copy()

    # Merge confidence values
    if text_df is not None:
        text_merged = pd.merge(main_df[['sample_id']],
                               text_df[['sample_id', 'confidence']],
                               on='sample_id', how='left')
        result_df['text_confidence'] = text_merged['confidence'].fillna(0.5)
    else:
        result_df['text_confidence'] = np.where(
            result_df['text_predicted_label'] == result_df['true_label'], 0.8, 0.2)

    if audio_df is not None:
        audio_merged = pd.merge(main_df[['sample_id']],
                                audio_df[['sample_id', 'confidence']],
                                on='sample_id', how='left')
        result_df['audio_confidence'] = audio_merged['confidence'].fillna(0.5)
    else:
        result_df['audio_confidence'] = np.where(
            result_df['audio_predicted_label'] == result_df['true_label'], 0.8, 0.2)

    if video_df is not None:
        video_merged = pd.merge(main_df[['sample_id']],
                                video_df[['sample_id', 'confidence']],
                                on='sample_id', how='left')
        result_df['video_confidence'] = video_merged['confidence'].fillna(0.5)
    else:
        result_df['video_confidence'] = np.where(
            result_df['video_predicted_label'] == result_df['true_label'], 0.8, 0.2)

    return result_df


def prepare_comparison_data(merged_df):
    """Prepare data for model comparison visualization"""

    text_vs_full_input = pd.DataFrame({
        'sample_id': merged_df['sample_id'],
        'unimodal_confidence': merged_df['text_confidence'],
        'full_input_confidence': merged_df['fusion_confidence'],
        'predictions_agree': merged_df['text_predicted_label'] == merged_df['fusion_predicted_label'],
        'both_correct': (merged_df['text_predicted_label'] == merged_df['true_label']) &
        (merged_df['fusion_predicted_label'] == merged_df['true_label'])
    })

    audio_vs_full_input = pd.DataFrame({
        'sample_id': merged_df['sample_id'],
        'unimodal_confidence': merged_df['audio_confidence'],
        'full_input_confidence': merged_df['fusion_confidence'],
        'predictions_agree': merged_df['audio_predicted_label'] == merged_df['fusion_predicted_label'],
        'both_correct': (merged_df['audio_predicted_label'] == merged_df['true_label']) &
        (merged_df['fusion_predicted_label'] == merged_df['true_label'])
    })

    video_vs_full_input = pd.DataFrame({
        'sample_id': merged_df['sample_id'],
        'unimodal_confidence': merged_df['video_confidence'],
        'full_input_confidence': merged_df['fusion_confidence'],
        'predictions_agree': merged_df['video_predicted_label'] == merged_df['fusion_predicted_label'],
        'both_correct': (merged_df['video_predicted_label'] == merged_df['true_label']) &
        (merged_df['fusion_predicted_label'] == merged_df['true_label'])
    })

    return text_vs_full_input, audio_vs_full_input, video_vs_full_input


def get_point_color_simple(agree, both_correct):
    """Simple color coding: focus on agreement and correctness"""
    if agree and both_correct:
        return '#2E8B57', 'Agree & Both Correct'  # Dark green
    elif agree and not both_correct:
        return '#4169E1', 'Agree & At Least One Wrong'  # Royal blue
    elif not agree and both_correct:
        return '#DC143C', 'Disagree & Both Correct'  # Crimson (rare case)
    else:
        return '#FF6347', 'Disagree & At Least One Wrong'  # Tomato


def create_simple_scatter_plot(ax, data, modality_name):
    """Create simplified scatter plot focusing on agreement"""

    # Assign colors
    colors = []
    labels = []
    for _, row in data.iterrows():
        color, label = get_point_color_simple(
            row['predictions_agree'], row['both_correct'])
        colors.append(color)
        labels.append(label)

    # Create scatter plot
    scatter = ax.scatter(data['unimodal_confidence'], data['full_input_confidence'],
                         c=colors, alpha=0.7, s=30, edgecolors='white', linewidth=0.5)

    # Add diagonal line (perfect agreement line)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5,
            linewidth=1, label='Perfect Agreement')

    # Set labels and limits
    ax.set_xlabel(f'{modality_name} Model Confidence')
    ax.set_ylabel('Full-Input Model Confidence')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    # ax.set_title(f'{modality_name} vs Full-Input', fontsize=12, pad=20)

    # Calculate statistics
    total = len(data)
    agree_rate = data['predictions_agree'].mean()
    both_correct_rate = data['both_correct'].mean()

    agree_and_correct = len(
        data[(data['predictions_agree']) & (data['both_correct'])])
    agree_not_correct = len(
        data[(data['predictions_agree']) & (~data['both_correct'])])
    disagree_correct = len(
        data[(~data['predictions_agree']) & (data['both_correct'])])
    disagree_not_correct = len(
        data[(~data['predictions_agree']) & (~data['both_correct'])])

    # # Add statistics text
    # stats_text = f"Total: {total}\n"
    # stats_text += f"Agreement: {agree_rate*100:.1f}%\n"
    # stats_text += f"Both Correct: {both_correct_rate*100:.1f}%\n\n"
    # stats_text += f"Agree+Correct: {agree_and_correct}\n"
    # stats_text += f"Agree+Wrong: {agree_not_correct}\n"
    # stats_text += f"Disagree+Correct: {disagree_correct}\n"
    # stats_text += f"Disagree+Wrong: {disagree_not_correct}"

    # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
    #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
    #         fontsize=9)

    return agree_rate, both_correct_rate


def create_summary_bar_chart(fig, text_stats, audio_stats, video_stats):
    """Create a summary bar chart showing agreement rates"""

    # Create subplot for bar chart
    ax_bar = plt.subplot2grid((2, 3), (1, 0), colspan=3, fig=fig)

    modalities = ['Text vs Full-Input',
                  'Audio vs Full-Input', 'Video vs Full-Input']
    agreement_rates = [text_stats[0]*100,
                       audio_stats[0]*100, video_stats[0]*100]
    both_correct_rates = [text_stats[1]*100,
                          audio_stats[1]*100, video_stats[1]*100]

    x = np.arange(len(modalities))
    width = 0.35

    bars1 = ax_bar.bar(x - width/2, agreement_rates, width, label='Prediction Agreement Rate',
                       color='skyblue', alpha=0.8)
    bars2 = ax_bar.bar(x + width/2, both_correct_rates, width, label='Both Models Correct Rate',
                       color='lightcoral', alpha=0.8)

    ax_bar.set_ylabel('Percentage (%)')
    # ax_bar.set_title('Model Performance Summary')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(modalities)
    ax_bar.legend()
    ax_bar.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)


def calculate_disagreement_rates(df):
    """Calculate pairwise disagreement rates between modalities"""
    modalities = ['text_predicted_label', 'audio_predicted_label',
                  'video_predicted_label', 'fusion_predicted_label']
    names = ['Text', 'Audio', 'Video', 'Full-Input']

    print("\n=== Pairwise Disagreement Rates ===")

    for i in range(len(modalities)):
        for j in range(i+1, len(modalities)):
            disagreement_rate = (df[modalities[i]] != df[modalities[j]]).mean()
            print(f"{names[i]} vs {names[j]}: {disagreement_rate:.3f}")


def main():
    # Load all prediction files
    print("Loading prediction files...")

    main_df = load_all_predictions('problem_define/fusion_all_predictions.csv')
    text_df = load_single_modality_predictions(
        'problem_define/text_predictions.csv')
    audio_df = load_single_modality_predictions(
        'problem_define/audio_predictions.csv')
    video_df = load_single_modality_predictions(
        'problem_define/video_predictions.csv')

    print(f"\nData overview:")
    print(f"Total samples: {len(main_df)}")
    print(f"Unique true labels: {sorted(main_df['true_label'].unique())}")

    # Calculate disagreement rates between modalities
    calculate_disagreement_rates(main_df)

    # Merge all prediction data
    print("\nMerging prediction data...")
    merged_df = merge_prediction_data(main_df, text_df, audio_df, video_df)

    # Prepare comparison data
    text_vs_full_input, audio_vs_full_input, video_vs_full_input = prepare_comparison_data(
        merged_df)

    # Create figure with cleaner layout
    fig = plt.figure(figsize=(15, 10))

    # Create scatter plots (top row)
    ax1 = plt.subplot2grid((2, 3), (0, 0), fig=fig)
    ax2 = plt.subplot2grid((2, 3), (0, 1), fig=fig)
    ax3 = plt.subplot2grid((2, 3), (0, 2), fig=fig)

    text_stats = create_simple_scatter_plot(ax1, text_vs_full_input, 'Text')
    audio_stats = create_simple_scatter_plot(ax2, audio_vs_full_input, 'Audio')
    video_stats = create_simple_scatter_plot(ax3, video_vs_full_input, 'Video')

    # Create summary bar chart (bottom row)
    create_summary_bar_chart(fig, text_stats, audio_stats, video_stats)

    # Create custom legend for scatter plots
    legend_elements = [
        Patch(facecolor='#2E8B57', label='Agree & Both Correct'),
        Patch(facecolor='#4169E1', label='Agree & At Least One Wrong'),
        Patch(facecolor='#DC143C', label='Disagree & Both Correct'),
        Patch(facecolor='#FF6347', label='Disagree & At Least One Wrong')
    ]

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95),
               ncol=4, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.4)

    # Save the plot
    plt.savefig('problem_define/model_prediction_consistency_simple.png',
                dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'problem_define/model_prediction_consistency_simple.png'")

    # Show the plot
    plt.show()

    # Print detailed statistics
    print("\n=== Model Consistency Analysis ===")
    modalities_data = [('Text', text_vs_full_input),
                       ('Audio', audio_vs_full_input), ('Video', video_vs_full_input)]

    for name, data in modalities_data:
        total = len(data)
        agree_rate = data['predictions_agree'].mean()
        both_correct_rate = data['both_correct'].mean()

        print(f"\n{name} vs Full-Input:")
        print(f"  Total samples: {total}")
        print(f"  Prediction agreement rate: {agree_rate*100:.1f}%")
        print(f"  Both models correct rate: {both_correct_rate*100:.1f}%")

    # Print model performance
    print("\n=== Individual Model Performance ===")
    text_acc = (merged_df['text_predicted_label']
                == merged_df['true_label']).mean()
    audio_acc = (merged_df['audio_predicted_label']
                 == merged_df['true_label']).mean()
    video_acc = (merged_df['video_predicted_label']
                 == merged_df['true_label']).mean()
    full_input_acc = (merged_df['fusion_predicted_label']
                      == merged_df['true_label']).mean()

    print(f"Text accuracy: {text_acc:.3f}")
    print(f"Audio accuracy: {audio_acc:.3f}")
    print(f"Video accuracy: {video_acc:.3f}")
    print(f"Full-Input accuracy: {full_input_acc:.3f}")


if __name__ == "__main__":
    main()
