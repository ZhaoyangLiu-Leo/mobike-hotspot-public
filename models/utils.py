import tensorflow as tf
import numpy as np
from util.plot_util import plot_embedding
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score

# Model construction utilities below adapted from
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#deep-mnist-for-experts


def dcg(predicted_order):
    i = 1
    cumulative_dcg = 0
    for x in predicted_order:
        cumulative_dcg += (2 ** x - 1) / (np.log(1 + i))
        i += 1
    return cumulative_dcg


def ndcg(predicted_order, top_count=100):
    sorted_list = np.sort(predicted_order)
    sorted_list = sorted_list[::-1]
    our_dcg = dcg(predicted_order[:top_count])
    if our_dcg == 0:
        return 0

    max_dcg = dcg(sorted_list[:top_count])
    ndcg_output = our_dcg / float(max_dcg)
    return ndcg_output


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            tmp_precision = num_hits / (i + 1.0)
            score += tmp_precision
            # print('Position {}, precision {}'.format(i + 1, tmp_precision))

    if len(actual) == 0:
        return 0.0

    return score / min(len(actual), k)


def map_metric(model_name, y_test, y_predict, percentile, top_k=10, hot_k=100, count_flag=True):
    y_test_df = pd.DataFrame({'id': np.arange(len(y_test)), 'value': y_test})
    y_test_df['hot'] = 0
    if count_flag:
        hot_threshold = (y_test_df['value'].sort_values(ascending=False)).iloc[hot_k]
        y_test_df.loc[y_test_df['value'] >= hot_threshold, 'hot'] = 1
    else:
        y_test_df.loc[y_test_df['value'] >= np.percentile(y_test, percentile), 'hot'] = 1

    y_pred_df = pd.DataFrame({'id': np.arange(len(y_predict)), 'value': y_predict})
    y_pred_df['hot'] = 0
    y_pred_df.loc[y_pred_df['value'] >= np.percentile(y_predict, percentile), 'hot'] = 1

    y_test_df.sort_values(by='value', ascending=False, inplace=True)
    y_pred_df.sort_values(by='value', ascending=False, inplace=True)

    ap_metric = apk(y_test_df[y_test_df['hot'] == 1].id.values, y_pred_df[y_pred_df['hot'] == 1].id.values, top_k)
    print('Model {}, MAP@{} metric {}'.format(model_name, top_k, ap_metric))
    return ap_metric


def ndcg_metric(model_name, y_test, y_predict, percentile, top_k=100, hot_k=100, count_flag=True):
    y_pred_df = pd.DataFrame({'id': np.arange(len(y_predict)), 'value': y_predict})
    y_pred_df['rel'] = 0
    if count_flag:
        hot_threshold = (np.sort(y_test))[-hot_k]
        y_pred_df.loc[y_test >= hot_threshold, 'rel'] = 1
    else:
        y_pred_df.loc[y_test >= np.percentile(y_test, percentile), 'rel'] = 1

    y_pred_df.sort_values(by='value', ascending=False, inplace=True)
    ndcg_val = ndcg(y_pred_df['rel'].values, top_k)

    print('Model {}, NDCG@{} metric {}'.format(model_name, top_k, ndcg_val))
    return ndcg_val


def recommend_evaluation(model_name, y_test, y_predict, percentile):
    print('========================================================================')
    ids = np.arange(len(y_test))
    test_hot_threshold = np.percentile(y_test, percentile)
    pred_hot_threshold = np.percentile(y_predict, percentile)
    index_true = ids[y_test >= test_hot_threshold]
    index_pred = ids[y_predict >= pred_hot_threshold]

    precision_k = len(set(index_pred).intersection(set(index_true))) / float(len(index_pred))
    recall_k = len(set(index_pred).intersection(set(index_true))) / float(len(index_true))
    jaccard = len(set(index_pred).intersection(set(index_true))) / float(len(set(index_pred).union(set(index_true))))

    print('Percentage %s, model recommend %s Performance:' % (percentile, model_name))
    print('Precision@Top %s %%: %s' % (100 - percentile, precision_k))
    print('Recall@Top %s %%: %s' % (100 - percentile, recall_k))
    print('Jaccard: %s' % jaccard)
    return precision_k, jaccard


def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    # return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = int(batch_count * batch_size)
        end = int(start + batch_size)
        batch_count += 1
        yield [d[start:end] for d in data]


def generate_combine_data(x_source, y_source, x_target, y_target, num_test=500):
    """
    generate test data
    include # of num_test source data and # of num_test target data
    """
    np.random.seed(1)
    x0, y0 = next(batch_generator([x_source, y_source], num_test))
    x1, y1 = next(batch_generator([x_target, y_target], num_test))

    combined_test_x = np.vstack([x0, x1])
    combined_test_labels = np.vstack([y0, y1])
    combined_test_domain = np.vstack([np.zeros((num_test, 1)),
                                      np.ones((num_test, 1))])
    return combined_test_x, combined_test_labels, combined_test_domain


def tsne_visualize(x, y, d, title):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    tsne_res = tsne.fit_transform(x)

    plot_embedding(tsne_res, y.squeeze(), d.squeeze(), title)


def hotspot_metric(y, y_pred):
    y_pred = np.round(y_pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(y, y_pred, average=None)
    return precision[1], recall[1], f_score[1]


def evaluate_rank(y, y_pred, percentile=90, top_k_s=(10, 30, 50), hot_count=100):
    metrics = dict()
    for top_k in top_k_s:
        metrics['map@{}'.format(top_k)] = map_metric('dann ranking', y.ravel(),
                                                     y_pred.ravel(), percentile, top_k=top_k, hot_k=hot_count)
        metrics['ndcg@{}'.format(top_k)] = ndcg_metric('dann ranking', y.ravel(),
                                                       y_pred.ravel(), percentile, top_k=top_k, hot_k=hot_count)

    for key, value in metrics.items():
        metrics[key] = round(value, 4)
    return metrics


def dann_evaluate(city_pair, task_type, source_metric, target_metric, domain_metric, y_target, target_pred, percent,
                  hot_count=100):
    print('Source (%s) metric:' % city_pair[0], source_metric)
    print('Target (%s) metric:' % city_pair[1], target_metric)
    print('Domain accuracy:', domain_metric)

    if task_type == 'reg':
        rank_metrics = evaluate_rank(y_target, target_pred, percent, hot_count=hot_count)
        rank_metrics['rmse'] = np.sqrt(np.mean(np.square(y_target.ravel() - target_pred.ravel())))
    else:
        rank_metrics = dict()
        rank_metrics['precision'] = precision_score(y_target.ravel(), np.round(target_pred.ravel()), average=None)[1]
        rank_metrics['recall'] = recall_score(y_target.ravel(), np.round(target_pred.ravel()), average=None)[1]
        rank_metrics['acc'] = target_metric
    return rank_metrics


def save_grid_search_res(save_path, search_res_list, columns, sort_col='map@10'):
    search_res_df = pd.DataFrame(search_res_list)
    if columns is not None:
        search_res_df = search_res_df[columns]
    search_res_df.sort_values(by=sort_col, ascending=False, inplace=True)
    search_res_df.to_csv(save_path, index=False)


def city_test(model, x_source, y_source, x_target, y_target, city_pair, train_evaluate_func,
              batch_size=64, num_steps=1000):
    graph = tf.get_default_graph()
    model.build_model()
    x_combine, y_combine, d_combine = generate_combine_data(x_source, y_source, x_target, y_target)

    print('\nSource only training')
    source_acc, target_acc, _, source_only_emb, source_pred, source_target_pred = train_evaluate_func(
        'source', graph, model, x_source, y_source,
        x_target, y_target, x_combine, d_combine, verbose=False, batch_size=batch_size, num_steps=num_steps)
    print('Source (%s) metric:' % city_pair[0], source_acc)
    # p, r, f = hotspot_metric(y_source.squeeze(), source_pred.squeeze())
    # print('Source (%s) hotspot precision: %s, recall: %s, f1 score: %s' % (city_pair[0], p, r, f))
    print('Target (%s) metric:' % city_pair[1], target_acc)
    # p, r, f = hotspot_metric(y_target.squeeze(), target_pred.squeeze())
    # print('Target (%s) hotspot precision: %s, recall: %s, f1 score: %s' % (city_pair[1], p, r, f))
    source_metrics = evaluate_rank(y_target, source_target_pred)

    print('\nDomain adaptation training')
    source_acc, target_acc, d_acc, dann_emb, source_pred, dann_target_pred = train_evaluate_func(
        'dann', graph, model, x_source, y_source,
        x_target, y_target, x_combine, d_combine, verbose=False, batch_size=batch_size, num_steps=num_steps
    )
    print('Source (%s) metric:' % city_pair[0], source_acc)
    #     p, r, f = hotspot_metric(y_source.squeeze(), source_pred.squeeze())
    #     print('Source (%s) hotspot precision: %s, recall: %s, f1 score: %s' % (city_pair[0], p, r, f))
    print('Target (%s) metric:' % city_pair[1], target_acc)
    #     p, r, f = hotspot_metric(y_target.squeeze(), target_pred.squeeze())
    #     print('Target (%s) hotspot precision: %s, recall: %s, f1 score: %s' % (city_pair[1], p, r, f))
    dan_metrics = evaluate_rank(y_target, dann_target_pred)
    print('Domain accuracy:', d_acc)

    return (x_combine, source_only_emb, dann_emb, y_combine, d_combine,
            source_target_pred, dann_target_pred, source_metrics, dan_metrics)


def scale_x(x_train, x_test, scaler=StandardScaler()):
    if len(x_train.shape) == 1:
        x_train = x_train[:, np.newaxis]
        x_test = x_test[:, np.newaxis]

    assert len(x_train.shape) == 2
    scaler = scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, scaler
