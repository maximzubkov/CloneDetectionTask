### CloneDetectionTask

> Задача заключается в том, чтобы при помощи фреймворка [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval) оценить метрики качества модели [`code2seq`](https://github.com/tech-srl/code2seq) в задаче поиска клонов кода. Работу можно разделить на несколько основных этапов: 
> - Загрузка предобрабученной модели [`code2seq`](https://github.com/tech-srl/code2seq)
> - Установка фреймворка [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval)
> - Преобразование данных датасета `IJaDatset` в формат подходящий модели [`code2seq`](https://github.com/tech-srl/code2seq)
> - Извлечение векторных представлений кода из модели на данных из `IJaDatset`
> - Поиск клонов с помощью полученных векторных представлений и приведение результатов к виду, требуемому [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval) и расчет метрик качества полученной модели с помощью [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval)

> Разберемся с каждым из этапов поподробнее:
> #### Шаг 1: загрузка модели [`code2seq`](https://github.com/tech-srl/code2seq)
> - Для того, чтобы получить доступ к модели, необходимо выполнить команду:
> ```
> git clone https://github.com/tech-srl/code2seq
> cd code2seq
> ```
> - В результате мы получим доступ к файлам модели, следующим шагом необходимо скачать предобученную модель к себе на компьютер, для этого необходимо выполнить команды:
> ```
> wget https://s3.amazonaws.com/code2seq/model/java-large/java-large-model.tar.gz
> tar -xvzf java-large-model.tar.gz
> ```
> - Выполнив эти команды, мы получим новую папку `models`, в которой будут лежать файлы с предобученной моделью
> #### Шаг 2: загрузка фреймворка [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval)
> Процесс установки подробно описан в файле readme github репозитории [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval), приведем здесь лишь выдержку:
> - В первую очередь необходимо склонировать [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval) внутрь уже склонированного репозитория [`code2seq`](https://github.com/tech-srl/code2seq), это можно сделать, находясь в папке code2seq и применив команды:
> ```
> git clone https://github.com/jeffsvajlenko/BigCloneEval
> cd BigCloneEval
> ```
> - Следующим шагом необходимо скачать последнюю версию BigCloneBench и IJaDataset, ссылки на скачивание:
> `Database, BigCloneEval Version: https://www.dropbox.com/s/z2k8r78l63r68os/BigCloneBench_BCEvalVersion.tar.gz?dl=0`
> `IJaDataset, BigCloneEval Version: https://www.dropbox.com/s/xdio16u396imz29/IJaDataset_BCEvalVersion.tar.gz?dl=0`
> Скачав файлы необходимо разархивировать файлы и перенести их из `IJaDataset` в папку `ijadataset/bcb_reduced/` , а`.psql` файл из `BigCloneEval` в папку `bigclonebenchdb`
> - Наконец необходимо выполнить следующую команду из папки `BigCloneEval`, чтобы скомпилировать код:
> ```
> make
> ```
> - Следом необходимо запустить скрипт `init` из директории `commands/`, она проинициализирует все необходимые командны для работы с базой данных
> #### Шаг 3: препроцессинг данных `IJaDataset`
> - За препроцессинг отвечает скрипт `preprocessing.sh` из папки `code2seq`. Из-за ограниченных возможностей моего компьютера, у меня не получилось сделать препроцессинг всех данных `ijadataset/bcb_reduced/`, поэтому я поступил следующим образом: выбрал некоторые папки из `ijadataset/bcb_reduced/` и скопировал их в папку `ijadataset/test/`, а также создал пустые папки `ijadataset/train/`, `ijadataset/val/` (это необходимо для корректной работы `preprocessing.sh`). Далее заменил поля файла `preprocessing.sh` следующим образом:
> ```
> TRAIN_DIR=BigCloneEval/ijadataset/train
> VAL_DIR=BigCloneEval/ijadataset/val
> TEST_DIR=BigCloneEval/ijadataset/test
> DATASET_NAME=BCE
> ```
> - Следующим шагом необходимо запустить препроцессинг командой:
> ```
> sh preprocessing.sh
> ```
> - После завершения работы скрипта `preprocessing.sh` предобработанные файлы будут лежать в файле `data/BCE/BCE.test.c2s`
> #### Шаг 4: извлечение векторных представлений данных
> - Чтобы получить векторные представления, воспользуемся функцией `evaluate` модуля `model.py`. Так как код файла `model.py` довольно большой, а изменения, которые я сделал достаточно малы, я поступил следующим образом: сначала я сделал `commit` с исходной версией `model.py`, а следом сделал `commit` с изменениями (название `Extracting embeddings in pickle format`), на всякий случай последовательно опишем сделанные в файле `model.py` изменения:
>   * Из функции `build_test_graph` будем возвращать еще одно значение -- `outputs.rnn_output`, строчка `590` файл `model.py`. 
>   * Далее в строках `153` и `600`, где используется функция `build_test_graph`, добавим через запятую новое возвращаемое значение.
>   * Теперь, чтобы получить значения возвращенного тензора, изменим строки `187-189`. Кроме того введем счетчик `batch_index`, чтобы сохранять эмбеддинги в разные файлы:
> ```
> predicted_indices, true_target_strings, top_values, embeddings = self.sess.run([self.eval_predicted_indices_op,
>                                                                                 self.eval_true_target_strings_op,
>                                                                                 self.eval_topk_values, 
>                                                                                 self.embeddings])
> np.save(f"save_{batch_index}", embeddings)
>   ```
>   * Аналогично поступим со строками `614-618`:
> ```
> predicted_indices, top_scores, \
> true_target_strings, attention_weights, \
> path_source_string, path_strings, path_target_string, \
> embeddings = self.sess.run([self.predict_top_indices_op, self.predict_top_scores_op, 
>                             self.predict_target_strings_op, self.attention_weights_op,
>                             self.predict_source_string, self.predict_path_string, 
>                             self.predict_path_target_string, self.embeddings],
>                             feed_dict={self.predict_placeholder: line})
> np.save("save", embeddings)
> ```
>   * Наконец, чтобы получить все векторные представления `ijadataset/test/`, запустим следующий код:
> ```
> python3 code2seq.py --load models/java-large-model/model_iter52.release --test data/BCE/BCE.test.c2s
> ```
> #### Шаг 5: оценка близости векторных представлений и подсчет метрик качества с помощью фреймворка [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval)
> - Теперь необходимо из имеющихся эмбеддингов выделить клонов. Можно посмотреть на задачу под разными угломи:
>   * Для начала стоит банально воспользоваться 1-Nearest-Neighbours по метрике Cosine Simularity
>   * Можно также попробовать обучить модель, предсказывающую является ли данная пара клонами друг друга. Тогда необходимо разделить выборку, предоставленную [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval) на `train`, `test`, `val` и обучить на ней модель. Затем следует применить методы фреймворка [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval) на выборке `test` (и строго на ней).
> - Наконец чтобы подсчитать метрики качетсва, необходимо знать на какой строке начинается и на какой заканчивается функция. Чтобы извлечь такую информацию, необходимо разбираться в скрипте `preprocessing.sh` и в файле `preprocessing.py`. В них происходит рекурсивное прочтение всех файлов и их токенизация функций. Формат [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval) требует, чтобы каждой функции соотвествовали значения ее первой и последней строки в коде. Чтобы для каждой функции запомнить строки, где функция начинается и где заканчивается, необходимо существенно менять код `preprocessing.py`. Я не успел этого сделать. Если предоставить данные пары клонов в правильном формате, то с помощью [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval) можно легко вычислить требующиеся метрики. Как именно это делать описано в `readme` [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval)
