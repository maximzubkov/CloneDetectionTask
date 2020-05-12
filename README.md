### CloneDetectionTask

> Задача заключается в том, чтобы при помощи фреймворка [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval) оценить метрики качества модели [`code2seq`](https://github.com/tech-srl/code2seq) в задаче поиска клонов кода. Работу можно разделить на несколько основных этапа: 
> - Загрузка предобрабученной модели [`code2seq`](https://github.com/tech-srl/code2seq)
> - Установка фреймворка [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval)
> - Преобразование данных датасета `IJaDatset` в формат подходящий модели [`code2seq`](https://github.com/tech-srl/code2seq)
> - Извлечение векторных представлений кода из модели на данных из `IJaDatset`
> - Поиск клонов с помощью полученных векторных представлений и приведение результатов к виду, требуемому [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval)
> - Расчет метрик качества полученной модели с помощью [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval)

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
> - Выполнив эту команду, мы получим новую папку `models`, в которой будут лежать файлы с предобученной моделью
> #### Шаг 2: загрузка фреймворка [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval)
> Процесс установки подробно описан в файле readme github репозитории [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval), приведем здесь лишь выдержку
> - В первую очередь необходимо склонировать [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval) внутрь уже склонированного репозитория [`code2seq`](https://github.com/tech-srl/code2seq), это можно сделать, находясь в папке code2seq и применив команды:
> ```
> git clone https://github.com/jeffsvajlenko/BigCloneEval
> cd BigCloneEval
> ```
> - Следующим шагом неодимо скачать последнюю версию BigCloneBench и IJaDataset, ссылки на скачиване:
> `Database, BigCloneEval Version: https://www.dropbox.com/s/z2k8r78l63r68os/BigCloneBench_BCEvalVersion.tar.gz?dl=0`
> `IJaDataset, BigCloneEval Version: https://www.dropbox.com/s/xdio16u396imz29/IJaDataset_BCEvalVersion.tar.gz?dl=0`
> Скачав файлы необходимо разорхивировать файлы и перенести файлы из `IJaDataset` в папку `ijadataset/bcb_reduced/` , а`.psql` файл из `BigCloneEval` в папку `bigclonebenchdb`
> - Наконец небходимо следующую команду из папки `BigCloneEval`, чтобы скомпилировать код:
> ```
> make
> ```
> - Следом необходимо запустить скрипт `init` из дикректории `commands/`, это команда проинициализирует все необходимые комнадны для работы с базой данных
> #### Шаг 3: препроцессинг данных `IJaDataset`
> - За препроцессинг ответсвенен скрипт `preprocessing.sh`, применить его к данным `IJaDataset` я поступил следующим образом: из-за ограниченных возможностей моего компьютера, у меня не получилось сделать препроцессинг всех данных `ijadataset/bcb_reduced/`, поэтому я выбрал некоторые папки из `ijadataset/bcb_reduced/` и скопировал их в папку `ijadataset/test/`, а также создал пустые папки `ijadataset/train/`, `ijadataset/val/` (это необходимо для корректной работы `preprocessing.sh`). Далее заменим поля файла `preprocessing.sh` следующим образом:
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
> - После завершения препроцессинга в файле data/BCE/BCE.test.c2s
> #### Шаг 4: извлечение векторных представлений данных
> - Чтобы получить векторные представления, воспользуемся функцией `evaluate` модуля `model.py`. Так как код файла `model.py` довольно большой, я поступил следующим образом: сначала я сделал `commit` с исходной версией `model.py`, а следом сделал `commit` с добавлениями, на всякий случай последовательно опишем изменения в нем, которые необходимо сделать для того, чтобы получить векторные представления слов:
>   * Из функции `build_test_graph` будет возвращать еще одно значение, а именно `outputs.rnn_output`, строчка `model.py`. 
>   * Далее необходимо во всех местах, где используется функция `build_test_graph` добавим через запятую еще одно возращаемое значение, а именно в строках `153` и `600` добавить `self.embeddings`
>   * Теперь, чтобы получить значения тензора, изменим строки `187-189`. Кроме того введем счетчик `batch_index`, чтобы сохранять эмбеддинги в разные файлы:
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
>   * Наконец, чтобы получить все векторные представления `dijadataset/test/`, запустим следующий код:
> ```
> python3 code2seq.py --load models/java-large-model/model_iter52.release --test data/BCE/BCE.test.c2s
> ```
> #### Шаг 5-6: оценка близости векторных представлений и подсчет метрик качества с помощью фреймворка [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval)
> - Чтобы подсчитать метрики качетсва необходимо знать на какой строке начинается и на какой заканчивается функция, чтобы извлечь такую информацию, необходимо разбираться в скрипте `preprocessing.sh` файле `preprocessing.py`. В них происходит рекурсивно прочитение всех файлов той или иной папки и затем парсинг и токнизация функций. Чтобы запомниать строки, где та или иная функция начинается и где заканчивается, необходимо существенно менять код `preprocessing.py`. Я к сожалению не успел этого сделать. Далее, если привести результаты работы модели (а именно пары функций являющихся клонами с указанием строк начла аи конца реализация каждой из функций), то с помощью [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval) можно легко вычислить метрики. Как именно это делать описано в `readme` [`BigCloneBench`](https://github.com/jeffsvajlenko/BigCloneEval)
