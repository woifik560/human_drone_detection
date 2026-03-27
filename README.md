# Описание
Проект для детекции людей с clover 4

# Требования
## Поддерживаемая ОПЕРАЦИОННАЯ СИСТЕМА
* 64-разрядная версия Raspberry Pi OS Bookworm
## Поддерживаемые платы
* Raspberry Pi 3 Model A+
* Raspberry Pi 3 Model B+
* Raspberry Pi 4 Model B

# Установка и настройка
* Создание виртуального окружения
```commandline
python -m venv venv
```
* Активация виртуального окружения
```commandline
source venv/bin/activate
```
* Установка OpenCV
```commandline
wget https://github.com/prepkg/opencv-raspberrypi/releases/latest/download/opencv_64.deb
sudo apt install -y ./opencv_64.deb
```
