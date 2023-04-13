# Распознавание телефонных номеров на фотографии
Была использована MobileNet V2 FPNLite с переносом обучения на один распознаваемый класс - телефонные номера.
Для того, чтобы обучить сеть на своих данных, нужно создать директорию images в основной директории проекта:
```
mkdir ./images
```
и загрузить туда пары файлов .jpg (изображения 640x640 пикселей) и .xml (разметка изображений в формате Pascal VOC XML) с одинаковыми названиями.

Затем, в соответствии с описанными в разметке классами, изменить файл по пути
> ./annotations/object-detection.pbtxt

и создать файлы TFRecord (последние 2 команды есть в файле generate_tfrecord.bat):
```
python xml-to-csv.py
python generate_tfrecord.py --csv_input=annotations/train_labels.csv --output_path=annotations/train.record --image_dir=images/train --labels_path=annotations/object-detection.pbtxt
python generate_tfrecord.py --csv_input=annotations/test_labels.csv --output_path=annotations/test.record --image_dir=images/test --labels_path=annotations/object-detection.pbtxt
```
Далее следует изменить файл 
> ./custom_models/your_model_name/pipeline.config

Основные теги для изменения: num_classes, batch_size, learning_rate_base, warmup_learning_rate, total_steps, num_steps, fine_tune_checkpoint, fine_tune_checkpoint_type, label_map_path, input_path

Запуск обучения (start_train.bat):
```
python model_main_tf2.py --model_dir=custom_models/ssd_mobilenet_v2_phone_numbers --pipeline_config_path=custom_models/ssd_mobilenet_v2_phone_numbers/pipeline.config
```

Экспорт модели (export.bat):
```
python exporter_main_v2.py --input_type=image_tensor --pipeline_config_path=custom_models\ssd_mobilenet_v2_phone_numbers\pipeline.config --trained_checkpoint_dir=custom_models\ssd_mobilenet_v2_phone_numbers --output_directory=exported_models\ssd_mobilenet_v2_phone_numbers
```

Пример локализации телефонного номера можно увидеть, запустив detection_test.ipynb
![image](https://user-images.githubusercontent.com/88054444/229306511-326845ca-5cf0-412c-addd-708a52a330cf.png)

TODO: распознавание отдельных цифр номера для получения его в виде строки (сделано в версии 1.1)
