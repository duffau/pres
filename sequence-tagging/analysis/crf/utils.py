from collections import Counter

def tag_distribution(dataset, sort="freq"):
    tag_names = dataset.features["ner_tags"].feature.names
    train_tags = dataset['ner_tags']
    flattened_tags = [tag for sublist in train_tags for tag in sublist]
    tag_counts = Counter(flattened_tags)
    tag_counts = {tag_names[tag_id]: counts for tag_id, counts in tag_counts.items()}
    if sort == "freq":
        tag_counts = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    elif sort == "alpha":
        tag_counts = sorted(tag_counts.items(), key=lambda x: x[0], reverse=False)
        tag_counts = sorted(tag_counts, key=lambda x: x[0][2:], reverse=False)
    else:
        raise ValueError("{sort} not recognized")
    return tag_counts


from datasets import Dataset
import random

def filter_sentences_by_tag(dataset: Dataset, tag_name: str) -> Dataset:
    # Get the index of the tag_name
    tag_index = dataset.features["ner_tags"].feature.names.index(tag_name)

    # Define a filter function that checks if the sentence contains the specified tag
    def filter_function(example):
        return tag_index in example['ner_tags']

    # Apply the filter function to the dataset
    filtered_dataset = dataset.filter(filter_function)

    return filtered_dataset


def format_sentences(tokens_list, tags_list):
    for tokens, tags in zip(tokens_list, tags_list):
        formatted_sentence = []
        for token, tag in zip(tokens, tags):
            if tag != "O":
                formatted_sentence.append(f"[{token}]{tag}")
            else:
                formatted_sentence.append(token)
        yield ' '.join(formatted_sentence)    

def format_sentence_examples(dataset: Dataset, tag_name: str, seed: int = None, n_sentences: int = 5) -> str:
    # Filter the dataset
    filtered_dataset = filter_sentences_by_tag(dataset, tag_name)

    # If seed is provided, set the random seed
    if seed is not None:
        random.seed(seed)

    # Randomly select n_sentences from the filtered dataset
    random_examples = random.sample(list(filtered_dataset), min(n_sentences, len(filtered_dataset)))

    # Extract tokens and tags for formatting
    tokens_list = [example['tokens'] for example in random_examples]
    tags_list = [[dataset.features["ner_tags"].feature.names[tag] for tag in example['ner_tags']] for example in random_examples]

    # Format the sentences
    return format_sentences(tokens_list, tags_list)

