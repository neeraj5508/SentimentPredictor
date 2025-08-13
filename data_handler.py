"""
Data handler for sentiment analysis application
Provides sample dataset for training the model
"""

def get_sample_dataset():
    """
    Returns a sample dataset with positive and negative sentiment examples
    
    Returns:
        list: List of dictionaries with 'text' and 'sentiment' keys
    """
    
    # Sample dataset with labeled sentiments (0: negative, 1: positive)
    sample_data = [
        # Positive examples
        {"text": "I absolutely love this product! It exceeded my expectations.", "sentiment": 1},
        {"text": "This is amazing! Best purchase I've made this year.", "sentiment": 1},
        {"text": "Fantastic quality and excellent customer service.", "sentiment": 1},
        {"text": "I'm so happy with this decision. Highly recommend!", "sentiment": 1},
        {"text": "Outstanding performance and great value for money.", "sentiment": 1},
        {"text": "Perfect! Everything works exactly as described.", "sentiment": 1},
        {"text": "This made my day! Thank you so much.", "sentiment": 1},
        {"text": "Excellent experience from start to finish.", "sentiment": 1},
        {"text": "I'm thrilled with the results. Five stars!", "sentiment": 1},
        {"text": "Beautiful design and incredible functionality.", "sentiment": 1},
        {"text": "This is exactly what I was looking for. Perfect!", "sentiment": 1},
        {"text": "Amazing quality at a reasonable price.", "sentiment": 1},
        {"text": "I love how easy this is to use.", "sentiment": 1},
        {"text": "Great job! This really impressed me.", "sentiment": 1},
        {"text": "Wonderful product with outstanding features.", "sentiment": 1},
        {"text": "This brings me so much joy every day.", "sentiment": 1},
        {"text": "Incredible value and superb quality.", "sentiment": 1},
        {"text": "I'm so grateful for this amazing experience.", "sentiment": 1},
        {"text": "This is the best thing ever! Love it!", "sentiment": 1},
        {"text": "Perfectly crafted and beautifully designed.", "sentiment": 1},
        {"text": "Absolutely fantastic! Will buy again.", "sentiment": 1},
        {"text": "Superb! Exceeded all my expectations.", "sentiment": 1},
        {"text": "The best service I have ever received.", "sentiment": 1},
        {"text": "I am extremely satisfied with my purchase.", "sentiment": 1},
        {"text": "Five stars! Highly recommended.", "sentiment": 1},
        {"text": "Everything was perfect from start to finish.", "sentiment": 1},
        {"text": "I couldn't be happier with this product.", "sentiment": 1},
        {"text": "Top-notch quality and performance.", "sentiment": 1},
        {"text": "A wonderful experience overall.", "sentiment": 1},
        {"text": "This is my favorite purchase this year.", "sentiment": 1},
        {"text": "The team was very helpful and responsive.", "sentiment": 1},
        {"text": "I will definitely recommend this to my friends.", "sentiment": 1},
        {"text": "The product works flawlessly.", "sentiment": 1},
        {"text": "I am delighted with the results.", "sentiment": 1},
        {"text": "A+ quality and service.", "sentiment": 1},
        {"text": "I am so pleased with this purchase.", "sentiment": 1},
        {"text": "The best investment I've made.", "sentiment": 1},
        {"text": "Everything was as described and more.", "sentiment": 1},
        {"text": "I am very impressed with the quality.", "sentiment": 1},
        {"text": "This exceeded my wildest expectations.", "sentiment": 1},
        
        # Negative examples
        {"text": "This is terrible! Complete waste of money.", "sentiment": 0},
        {"text": "I hate this product. It doesn't work at all.", "sentiment": 0},
        {"text": "Awful quality and poor customer service.", "sentiment": 0},
        {"text": "Very disappointed with this purchase.", "sentiment": 0},
        {"text": "This is the worst experience I've ever had.", "sentiment": 0},
        {"text": "Completely broken and useless. Don't buy this!", "sentiment": 0},
        {"text": "I regret buying this. It's horrible.", "sentiment": 0},
        {"text": "Poor quality and overpriced. Not worth it.", "sentiment": 0},
        {"text": "This ruined my day. Extremely frustrating.", "sentiment": 0},
        {"text": "Terrible design and doesn't work properly.", "sentiment": 0},
        {"text": "I'm very unhappy with this decision.", "sentiment": 0},
        {"text": "This is frustrating and poorly made.", "sentiment": 0},
        {"text": "Disappointing results and bad experience.", "sentiment": 0},
        {"text": "I feel cheated. This is not as advertised.", "sentiment": 0},
        {"text": "Worst purchase ever. Completely useless.", "sentiment": 0},
        {"text": "This makes me angry. So poorly designed.", "sentiment": 0},
        {"text": "I'm disgusted with the quality of this.", "sentiment": 0},
        {"text": "Horrible experience and terrible support.", "sentiment": 0},
        {"text": "This is a disaster. Nothing works right.", "sentiment": 0},
        {"text": "I'm extremely dissatisfied with everything.", "sentiment": 0},
        {"text": "Absolutely horrible. Would not recommend.", "sentiment": 0},
        {"text": "The worst product I have ever used.", "sentiment": 0},
        {"text": "Completely unsatisfactory and disappointing.", "sentiment": 0},
        {"text": "I am very upset with this purchase.", "sentiment": 0},
        {"text": "Nothing worked as expected. Terrible.", "sentiment": 0},
        {"text": "Customer service was unhelpful and rude.", "sentiment": 0},
        {"text": "I wasted my money on this.", "sentiment": 0},
        {"text": "Extremely poor quality and performance.", "sentiment": 0},
        {"text": "I will never buy this again.", "sentiment": 0},
        {"text": "The product broke after one use.", "sentiment": 0},
        {"text": "I am very disappointed and frustrated.", "sentiment": 0},
        {"text": "This is not worth the price at all.", "sentiment": 0},
        {"text": "I had a terrible experience overall.", "sentiment": 0},
        {"text": "The quality is shockingly bad.", "sentiment": 0},
        {"text": "I regret this purchase completely.", "sentiment": 0},
        {"text": "The worst investment I've made.", "sentiment": 0},
        {"text": "Nothing good to say about this.", "sentiment": 0},
        {"text": "I am dissatisfied in every way.", "sentiment": 0},
        {"text": "This is a complete letdown.", "sentiment": 0},
        {"text": "I wish I could get my money back.", "sentiment": 0},
        
        # Mixed/Neutral examples for better training
        {"text": "It's okay, nothing special but does the job.", "sentiment": 0},
        {"text": "Good value for the price, would recommend.", "sentiment": 1},
        {"text": "Not bad, but could be better in some areas.", "sentiment": 0},
        {"text": "Pretty good overall, happy with my choice.", "sentiment": 1},
        {"text": "Mediocre quality, expected more for the price.", "sentiment": 0},
        {"text": "Pleasant surprise! Better than I expected.", "sentiment": 1},
        {"text": "Decent product but has some issues.", "sentiment": 0},
        {"text": "Really impressed with the quick delivery.", "sentiment": 1},
        {"text": "The interface is confusing and hard to use.", "sentiment": 0},
        {"text": "Smooth operation and user-friendly design.", "sentiment": 1},
    ]
    
    return sample_data

def get_dataset_statistics():
    """
    Returns basic statistics about the sample dataset
    
    Returns:
        dict: Dictionary with dataset statistics
    """
    data = get_sample_dataset()
    
    total_samples = len(data)
    positive_samples = len([item for item in data if item['sentiment'] == 1])
    negative_samples = len([item for item in data if item['sentiment'] == 0])
    
    return {
        'total_samples': total_samples,
        'positive_samples': positive_samples,
        'negative_samples': negative_samples,
        'positive_ratio': positive_samples / total_samples,
        'negative_ratio': negative_samples / total_samples
    }
