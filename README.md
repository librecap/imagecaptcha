# ImageCaptcha

Generate highly secure, accessible image CAPTCHAs. Human recognizable but bot-resistant image challenges with sequences of characters, slider puzzle captchas and sophisticated obfuscation techniques. 

## Quick Start

```bash
git clone https://github.com/librecap/imagecaptcha.git
cd imagecaptcha/sliding
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python generate_captcha.py
```

## Overview

ImageCaptcha is a Python library for generating highly secure, accessible image CAPTCHA challenges that are human recognizable but resistant to automated attacks. The library provides various types of visual challenges including character sequences (coming soon) and slider puzzles, with sophisticated obfuscation techniques to enhance security while maintaining usability. The CAPTCHAs are designed to be intuitive for humans to solve while incorporating multiple layers of bot-detection mechanisms.

This is a Prove of Concept implementation intended to be used for a Rust implementation for the LibreCap project.

## Features/Types

### Currently Available

#### Sliding Puzzle
- **Description**: A sliding puzzle captcha with a 3D cube
- **Customization**: Adjustable difficulty level and background patterns
- **Accessibility**: Visual puzzle piece and slider interface for intuitive interaction, with clear feedback on success/failure.

### Planned Features

#### Character Sequence
- The user has to identify the characters in an image.

## Installation

```bash
git clone https://github.com/librecap/imagecaptcha.git
cd imagecaptcha/sliding
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python generate_captcha.py
```

### Requirements
- Python 3.7+
- opencv-python
- numpy
- scipy

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/librecap/imagecaptcha.git
cd imagecaptcha

# Choose a captcha type
cd sliding

# Create a new virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
# You may need to use python -m ensurepip to install pip
pip install -r requirements.txt
```

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

Copyright 2025 LibreCap Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.