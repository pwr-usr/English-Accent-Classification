# English Accent Detection

This repository contains the code and resources for the project **"English Accent Detection: A Comparison of Audio, Spectrogram, and Text Classification Methods using Transformers"**. This project explores various methods to classify English accents using machine learning, specifically focusing on transformer architectures.

## Project Overview

### Abstract

Accent provides additional speech information beyond content and speaker identity. Accurate accent classification can improve speech recognition through accent-specific models and enhance speaker recognition. Understanding accents is also crucial for machine interactions with humans. This project uses three types of data inputs to classify eight English accents: audio arrays, spectrogram images, and text transcripts. The results demonstrate the effectiveness of transformer architectures for accent recognition when learning from audio data or spectrogram images. The developed models have applications in speech recognition, language learning assessment, and mitigating accent-based disparities in voice interface technologies.

### Methods and Models

We implemented three methods for accent classification:

1. **Audio Arrays**: Using the wav2vec 2.0 model, we processed raw audio arrays to classify accents.
2. **Spectrogram Images**: Audio data was converted into spectrogram images and classified using the Vision Transformer (ViT) model.
3. **Text Transcripts**: Transcripts of speech were classified using the BERT model.

## Dataset

The dataset contains 7680 rows of data, each comprising audio, audio transcripts, and an accent label. The accent labels include eight different classes:
- United States English
- India and South Asia
- England English
- Scottish English
- Irish English
- Canadian English
- Australian English
- Filipino

## Installation

To run the code, you need to install the following dependencies:

```bash
pip install datasets evaluate accelerate wandb transformers librosa scikit-learn matplotlib seaborn
```

## Usage

### Audio Array Classification

1. **Setup the environment**:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Load the data**:
    ```python
    import pyarrow.parquet as pq
    import pandas as pd
    data = pd.concat([pq.read_table(f'/content/drive/MyDrive/data/{i}.parquet').to_pandas() for i in range(8)])
    data.reset_index(drop=True, inplace=True)
    ```

3. **Train the model**:
    ```python
    from transformers import Wav2Vec2Processor, AutoModelForAudioClassification, TrainingArguments, Trainer
    feature_extractor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=8)
    ```

4. **Evaluate the model**:
    ```python
    trainer.evaluate()
    ```

### Spectrogram Classification

1. **Convert audio to spectrogram**:
    ```python
    import librosa
    def audio_to_melspectrogram(audio, sr=16000):
        melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        return librosa.power_to_db(melspec, ref=np.max)
    ```

2. **Prepare the dataset**:
    ```python
    from datasets import Dataset, Image
    train_dataset = Dataset.from_dict({"image": train_images_path}).cast_column("image", Image())
    ```

3. **Train the model**:
    ```python
    from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=8)
    ```

### Text Classification

1. **Tokenize the data**:
    ```python
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ```

2. **Prepare the dataset**:
    ```python
    def tokenize_function(example):
        return tokenizer(example["sentence"], padding="max_length")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    ```

3. **Train the model**:
    ```python
    from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=8)
    ```

## Results

- **Audio Array Model**: Achieved 99.8% accuracy on the test set.
- **Spectrogram Image Model**: Achieved 99.02% accuracy on the test set.
- **Text Model**: Performed poorly with around 20% accuracy.

## Conclusion

Transformers are highly effective for accent recognition when using audio data or spectrogram images, but text alone is insufficient for accurate accent classification.

## Acknowledgments

This work was done as part of the ST311 course at the London School of Economics and Political Science.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- [Colab Notebook](https://colab.research.google.com/drive/1V59vfCSLv6DxCXBrqy1M9k081DuEAcjB?usp=sharing)
- [Project Report](./11-paper.pdf)
- [Project Slides](./11-slides.pdf)

---

Feel free to clone this repository and explore the different methods used for accent detection!
