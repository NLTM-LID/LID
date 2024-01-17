# Spoken Language Identification using u-vector

This work is the 1st version of the spoken language identification (LID) task. It uses the pre-trained model based on the u-vector with Within-Sample Similarity Loss (WSSL). This model identifies the 11 Indian languages (Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu) and English. In this work, the following labels are used for these languages. 

- Assamese- asm
- Bengali- ben
- Gujarati- guj
- Hindi- hin
- Kannada- kan
- Malayalam- mal
- Marathi- mar
- Odia- odi
- Punjabi- pun
- Tamil- tam
- Telugu- tel
- English- eng

This repository contains the Python script of a graphical user interface (GUI) and the command line interface (CLI) for Spoken Language Identification using u-vector with WSSL model. The GUI allows users to perform various tasks such as recording audio, playing saved audio files, and identifying the language of spoken audio using u-vector with WSSL models. 

The CLI also identifies the language of spoken audio using u-vector with WSSL models, but it takes either a single audio file or a directory containing multiple audio files as input. 

Both tasks, GUI and CLI use the pre-trained models to identify the language of the spoken audio. The minimum duration of the spoken audio should be around 5-10 seconds.

## Note: Please follow the corresponding Readme.md file for more details.

# Limitation
These models are trained on speech datasets taken from the EkStep repository collected from TV/Radio news, YouTube videos, recordings, etc. These datasets do not cover all possible types of speech such as dialects, accents, channels, and domains. Therefore, these models may fail in the following conditions.

- Presence of unfamiliar dialects and accents.
- Presence of high domain mismatch.
- Contains too much noise and unclear speech.

# Acknowledgement

This work is performed with the support of the project named "Speech Technologies In Indian Languages". It is part of the NLTM (National Language Technology Mission) consortium project which is sponsored by Meity (Ministry of Electronics and Information Technology), India.


# License

This project is licensed under the NLTM License - see the [LICENSE](LICENSE) file for details.

