# -*- coding: utf-8 -*-

from models.flip_gradient import flip_gradient
from models.utils import *
from models.data_util import load_city_pair_data
from util.plot_util import plot_losses
import os
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.model_selection import train_test_split
import datetime
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

logger = logging.getLogger(__name__)


"""
Description:
    City domain adaptation model:
        1. Feature network component: multi-layer feed forward neural network;
        2. Density network component: regression loss;
        3. Domain network component: city domain adaption by the gradient reversal layer;
Author: Zhaoyang Liu
Created time: 2017-12-10
"""


class DaCityModel(object):
    """
    Mobike hotspot detection, city domain adaptation model.
    """

    def __init__(self, input_dim, output_dim=1, init_learning_rate=0.001, optimizer='adam',
                 batch_size=64, task_type='reg', pos_weight=3,
                 use_batch_norm=True,
                 feature_layers=(32, 32), feature_dim=64, feature_dropout=0.2,
                 predictor_layers=(32, 16), predictor_dropout=0.2,
                 domain_layers=(16,), beta=1, tensor_board=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_learning_rate = init_learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.task_type = task_type
        self.pos_weight = pos_weight
        self.use_batch_norm = use_batch_norm

        self.feature_layers = feature_layers
        self.feature_dim = feature_dim
        self.feature_dropout = feature_dropout

        self.predictor_layers = predictor_layers
        self.predictor_dropout = predictor_dropout

        self.domain_layers = domain_layers
        self.beta = beta

        self.tensor_board = tensor_board

        self.X = None
        self.y = None
        self.domain = None
        self.l = None
        self.dann = None
        self.train = None
        self.feature = None

        self.total_loss = None
        self.learning_rate = None
        self.regular_train_op = None
        self.dann_train_op = None

        self.domain_acc = None
        self.label_acc = None

        self.pred = None
        self.pred_loss = None

        # self.build_model()

    def init_variable(self):
        self.X = tf.placeholder(tf.float32, [None, self.input_dim])
        self.y = tf.placeholder(tf.float32, [None, self.output_dim])
        self.domain = tf.placeholder(tf.float32, [None, 1])
        self.l = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, name='train_mode')
        self.dann = tf.placeholder(tf.bool, name='data_mode')
        self.learning_rate = tf.placeholder(tf.float32, [])

    def _build_feature_net(self):
        """
        MLP model for feature extraction
        :return:
        """
        with tf.variable_scope('feature_extractor'):
            net = self.X
            for i, layer_unit in enumerate(self.feature_layers):
                with tf.variable_scope('layer{}'.format(i)):
                    net = tf.layers.dense(net, layer_unit, name='layer{}_fc'.format(i))
                    if self.use_batch_norm:
                        net = tf.layers.batch_normalization(net, training=self.train, name='layer{}_bn'.format(i))
                    net = tf.nn.relu(net)
                    if self.feature_dropout > 0:
                        net = tf.layers.dropout(net, rate=self.feature_dropout,
                                                training=self.train, name='layer{}_bn'.format(i))
            net = tf.layers.dense(net, self.feature_dim, name='output')
        return net

    def _build_predictor_net(self):
        """
        MLP for class prediction
        """
        with tf.variable_scope('label_predictor'):
            # Switches to route target examples (second half of batch) differently
            source_feats = tf.cond(
                self.dann,
                lambda: tf.slice(self.feature, [0, 0], [int(self.batch_size / 2), -1]),
                lambda: self.feature
            )

            self.source_labels = tf.cond(
                self.dann,
                lambda: tf.slice(self.y, [0, 0], [int(self.batch_size / 2), -1]),
                lambda: self.y
            )

            net = source_feats
            for i, layer_unit in enumerate(self.predictor_layers):
                with tf.variable_scope('layer{}'.format(i)):
                    net = tf.layers.dense(net, layer_unit)
                    if self.use_batch_norm:
                        net = tf.layers.batch_normalization(net, training=self.train)
                    net = tf.nn.relu(net)
                    if self.predictor_dropout > 0:
                        net = tf.layers.dropout(net, rate=self.predictor_dropout, training=self.train)

            pred = tf.layers.dense(net, self.output_dim)
        return pred

    def build_pred_loss(self, pred):
        if self.task_type == 'reg':
            self.pred = pred
            self.pred_loss = tf.reduce_mean(tf.square(self.pred - self.source_labels), name='pred_loss')
        else:
            self.pred = tf.nn.sigmoid(pred)
            self.pred_loss = tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(logits=pred, targets=self.source_labels,
                                                         pos_weight=self.pos_weight))

    def _build_domain_net(self):
        with tf.variable_scope('domain_predictor'):
            # Flip the gradient when back propagating through this operation
            feat = flip_gradient(self.feature, self.l)

            net = feat
            for i, layer_unit in enumerate(self.domain_layers):
                with tf.variable_scope('layer{}'.format(i)):
                    net = tf.layers.dense(net, layer_unit)

            d_logits = tf.layers.dense(net, 1)
            self.domain_pred = tf.nn.sigmoid(d_logits)
            self.domain_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, labels=self.domain), name='domain_loss')

    def build_model(self):
        tf.set_random_seed(1)

        self.init_variable()
        # The domain-invariant feature
        self.feature = self._build_feature_net()

        # MLP for class prediction
        pred = self._build_predictor_net()
        self.build_pred_loss(pred)

        # Small MLP for domain prediction with adversarial loss
        self._build_domain_net()
        self.total_loss = self.pred_loss + self.beta * self.domain_loss

        # optimizer
        if self.optimizer == 'adam':
            self.regular_train_op = tf.train.AdamOptimizer(self.init_learning_rate).minimize(self.pred_loss)
            self.dann_train_op = tf.train.AdamOptimizer(self.init_learning_rate).minimize(self.total_loss)
        else:
            self.regular_train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.pred_loss)
            self.dann_train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.total_loss)

        # acc evaluate
        correct_domain_pred = tf.equal(self.domain, tf.round(self.domain_pred))
        self.domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

        if self.task_type == 'cls':
            self.label_acc = self.pred_loss
            # correct_label_pred = tf.equal(self.source_labels, tf.round(self.pred))
            # self.label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
        else:
            self.label_acc = tf.sqrt(self.pred_loss)


class Solver:
    def __init__(self, sess, model):
        self.model = model
        self.sess = sess

    def train(self, x_c, y, l, lr):
        """
        batch training on source or target
        :return:
        """
        _, batch_loss = self.sess.run(
            [self.model.regular_train_op, self.model.pred_loss],
            feed_dict={
                self.model.X: x_c, self.model.y: y,
                self.model.dann: False, self.model.train: True,
                self.model.l: l,
                self.model.learning_rate: lr
            }
        )
        return batch_loss

    def train_dann(self, x_c, y, domain_labels, l, lr):
        """
        batch training on dann mode
        :return:
        """
        feed_dict = {
            self.model.X: x_c, self.model.y: y, self.model.domain: domain_labels,
            self.model.dann: True, self.model.train: True, self.model.l: l, self.model.learning_rate: lr
        }

        _, batch_loss, domain_loss, pred_loss, d_acc, p_acc = self.sess.run(
            [
                self.model.dann_train_op, self.model.total_loss, self.model.domain_loss,
                self.model.pred_loss, self.model.domain_acc, self.model.label_acc
            ],
            feed_dict=feed_dict
        )

        return batch_loss, domain_loss, pred_loss, d_acc, p_acc

    def evaluate(self, x, y, batch_size=None):
        if batch_size is not None:
            n = x.shape[0]
            acc = 0.0
            predicts = []

            for i in range(0, n, batch_size):
                x_batch = x[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                batch_acc, batch_pred = self.sess.run(
                    [self.model.label_acc, self.model.pred],
                    feed_dict={
                        self.model.X: x_batch, self.model.y: y_batch,
                        self.model.dann: False, self.model.train: False
                    }
                )
                acc += batch_acc * x_batch.shape[0]
                predicts.append(batch_pred)
            acc /= n
            pred = np.vstack(predicts)
        else:
            acc, pred = self.sess.run(
                [self.model.label_acc, self.model.pred],
                feed_dict={
                    self.model.X: x, self.model.y: y,
                    self.model.dann: False, self.model.train: False
                }
            )

        return acc, pred

    def evaluate_domain(self, x_combine, d_combine):
        domain_acc = self.sess.run(
            self.model.domain_acc,
            feed_dict={
                self.model.X: x_combine,
                self.model.domain: d_combine, self.model.l: 1.0,
                self.model.dann: False, self.model.train: False
            }
        )

        return domain_acc

    def predict(self, x):
        feed = {
            self.model.X1: x,
            self.model.dann: False,
            self.model.train: False
        }

        y_pred = self.model.pred
        return self.sess.run([y_pred], feed_dict=feed)


def train_and_evaluate_mlp(training_mode, graph, model, x_train, y_train, x_val, y_val, x_test, y_test,
                           combine_x, combine_d, verbose=False, batch_size=64, num_steps=4000, mode='mlp_s',
                           early_stop=True):
    """Helper to run the model with different training modes."""

    verbose_time = 100
    evaluate_time = 100
    source_losses = []
    val_losses = []
    target_losses = []

    best_val_losses = 1000.0
    early_stop_flag = 0

    saver = tf.train.Saver()
    check_point_path = os.path.join(
        '../check_point', '%s_%s' % (mode, datetime.datetime.now().strftime('%m%d_%H%M%S')))
    if not os.path.exists(check_point_path):
        os.mkdir(check_point_path)
    save_path = os.path.join(check_point_path, 'model.ckpt')

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        solver = Solver(sess=sess, model=model)

        # Batch generators
        gen_source_batch = batch_generator(
            [x_train, y_train], batch_size // 2)
        gen_target_batch = batch_generator(
            [x_test, y_test], batch_size // 2)
        gen_source_only_batch = batch_generator(
            [x_train, y_train], batch_size)
        gen_target_only_batch = batch_generator(
            [x_test, y_test], batch_size)

        domain_labels = np.vstack([np.zeros((batch_size // 2, 1)),
                                   np.ones((batch_size // 2, 1))])

        # Training loop
        for i in range(num_steps):
            # Adaptation param and learning rate schedule as described in the paper
            p = float(i) / num_steps
            l = 2. / (1. + np.exp(-10. * p)) - 1
            # lr = 0.01 / (1. + 10 * p) ** 0.75  # learning rate decay
            lr = 0.0001

            # Training step
            if training_mode == 'dann':
                x_s, y_s = next(gen_source_batch)
                x_t, y_t = next(gen_target_batch)
                x_c = np.vstack([x_s, x_t])
                y = np.vstack([y_s, y_t])

                batch_loss, domain_loss, pred_loss, d_acc, p_acc = solver.train_dann(x_c, y, domain_labels, l, lr)

                if verbose and i % verbose_time == 0:
                    print('step: %d, loss: %f  d_acc: %f  p_acc: %f  p: %f  l: %f  lr: %f' % (
                        i, batch_loss, d_acc, p_acc, p, l, lr))

            elif training_mode == 'source':
                x_c, y = next(gen_source_only_batch)
                batch_loss = solver.train(x_c, y, l, lr)
                if verbose and i % verbose_time == 0:
                    print('step: %d, source loss: %f' % (i, batch_loss))

            elif training_mode == 'target':
                x_c, y = next(gen_target_only_batch)
                batch_loss = solver.train(x_c, y, l, lr)
                if verbose and i % verbose_time == 0:
                    print('step: %d, target loss: %f' % (i, batch_loss))

                    # Compute final evaluation on test data
            if i % evaluate_time == 0:
                train_acc, _ = solver.evaluate(x_train, y_train, batch_size)
                val_acc, _ = solver.evaluate(x_val, y_val, batch_size)
                test_acc, _ = solver.evaluate(x_test, y_test, batch_size)
                test_domain_acc = solver.evaluate_domain(combine_x, combine_d)
                print('step: %s, source train metric: %f, source val metric: %f, target metric: %f, domain acc: %f' % (
                    i, train_acc, val_acc, test_acc, test_domain_acc))

                if val_acc < best_val_losses:
                    logger.info('Update the model, last best loss: %f, current best loss: %f' % (
                        best_val_losses, val_acc))
                    save_path = saver.save(sess, save_path)
                    best_val_losses = val_acc

                    early_stop_flag = 0
                else:
                    early_stop_flag += 1
                    if early_stop and early_stop_flag > 10:
                        break

                source_losses.append(train_acc)
                val_losses.append(val_acc)
                target_losses.append(test_acc)
        saver.restore(solver.sess, save_path)
        train_acc, source_pred = solver.evaluate(x_train, y_train, batch_size)
        val_acc, _ = solver.evaluate(x_val, y_val, batch_size)
        test_acc, target_pred = solver.evaluate(x_test, y_test, batch_size)
        test_domain_acc = solver.evaluate_domain(combine_x, combine_d)
        test_emb = sess.run(model.feature, feed_dict={model.X: combine_x, model.dann: False, model.train: False})

        source_losses.append(train_acc)
        val_losses.append(val_acc)
        target_losses.append(test_acc)
        plot_losses(source_losses, target_losses, os.path.join(
            '../results', 'dann_' + training_mode + '_loss.png'))

    return train_acc, val_acc, test_acc, test_domain_acc, test_emb, source_pred, target_pred, save_path


def evaluate(dann_model, solver_class, model_path, x_test, y_test, batch_size=64, hot_count=100):
    """
    evaluate the best parameter
    :return:
    """
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    dann_model.build_model()

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=graph) as sess:
        solver = solver_class(sess=sess, model=dann_model)
        saver.restore(solver.sess, model_path)
        test_acc, target_pred = solver.evaluate(x_test, y_test, batch_size)
    rank_metrics = evaluate_rank(y_test, target_pred, hot_count=hot_count)
    rank_metrics['rmse'] = test_acc
    return rank_metrics


def run_mlp(x_source, y_source, x_target, y_target, city_pair, train_mode='source', bs=128, train_num_steps=9001,
            init_learning_rate=0.001, percent=90, feature_dropout=0.2, predictor_dropout=0.2, beta=1, task_type='reg',
            pos_weight=1, use_batch_norm=True, hot_count=100):
    x_combine, y_combine, d_combine = generate_combine_data(x_source, y_source, x_target, y_target)
    x_train, x_val, y_train, y_val = train_test_split(x_source, y_source, test_size=0.1, random_state=42)

    dann_model = DaCityModel(input_dim=x_source.shape[-1], init_learning_rate=init_learning_rate, optimizer='adam',
                             tensor_board=False, batch_size=bs, use_batch_norm=use_batch_norm, task_type=task_type,
                             feature_dropout=feature_dropout, predictor_dropout=predictor_dropout, beta=beta,
                             pos_weight=pos_weight)
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    dann_model.build_model()

    print('\n%s only training' % train_mode)
    train_metric, val_metric, test_metric, domain_metric, feature_embedding, source_pred, target_pred, model_path = \
        train_and_evaluate_mlp(
            train_mode, graph, dann_model, x_train, y_train, x_val, y_val,
            x_target, y_target, x_combine, d_combine, verbose=False, batch_size=bs, num_steps=train_num_steps,
            mode='mlp_%s' % train_mode
        )

    rank_metrics = dann_evaluate(city_pair, task_type, train_metric, test_metric, domain_metric,
                                 y_target, target_pred, percent, hot_count=hot_count)

    return rank_metrics, val_metric, model_path


def run_mlp_one_time(train_mode='source', city_pair=('bj', 'nb'),
                     path_pattern='../data/road/train_bound_unique_new_filter/%s_500_week.csv', task_type='reg',
                     hot_count=100):
    percent = 90
    train_num_steps = 10001
    bs = 128
    init_learning_rate = 0.001
    feature_dropout = 0
    use_batch_norm = True
    beta = 1

    x_source, y_source, x_target, y_target = load_city_pair_data(
        city_pair[0], city_pair[1], path_pattern=path_pattern, percent=percent, task_type=task_type)

    rank_metrics, _, _ = run_mlp(x_source, y_source, x_target, y_target, city_pair, train_mode=train_mode,
                                 train_num_steps=train_num_steps, bs=bs, init_learning_rate=init_learning_rate,
                                 feature_dropout=feature_dropout, beta=beta, task_type=task_type,
                                 use_batch_norm=use_batch_norm, hot_count=hot_count)

    print(str(rank_metrics))
    return rank_metrics


def grid_search_mlp(train_mode='source', search_mode='grid', repeat=4, city_pair=('bj', 'nb'),
                    data_version='bound_unique_new', task_type='reg', percent=90, hot_count=100):
    x_source, y_source, x_target, y_target = load_city_pair_data(
        city_pair[0], city_pair[1],
        path_pattern='../data/road/train_' + data_version + '/%s_500_week.csv',
        percent=percent, task_type=task_type
    )

    train_num_steps = 20001
    best_val_metric = 1000.0
    best_model_path = None
    best_param = None

    param_grid = dict(
        bs=[64, 128, 256],
        init_learning_rate=[0.001, 0.0001],
        feature_dropout=[0.0, 0.2, 0.4],
    )
    if train_mode == 'dann':
        param_grid['beta'] = [0.01, 0.1, 1]
    else:
        param_grid['beta'] = [1]

    if task_type == 'cls':
        param_grid['pos_weight'] = [1, 3, 5]

    if search_mode == 'grid':
        candidate_params = list(ParameterGrid(param_grid))
    else:
        candidate_params = list(ParameterSampler(param_grid, n_iter=10))

    save_dir = '../model_search_res/%s/' % data_version
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    search_res_list = []
    for param in candidate_params:
        print(str(param))
        for i in range(repeat):
            run_dict = param.copy()
            run_dict['repeat'] = i
            rank_metrics, val_metric, model_path = run_mlp(
                x_source, y_source, x_target, y_target, city_pair, train_mode=train_mode,
                task_type=task_type, train_num_steps=train_num_steps, hot_count=hot_count, **param
            )
            run_dict.update(rank_metrics)

            if val_metric < best_val_metric:
                logger.info('Choice new parameter, last best val metric: %f, new best metric: %f' % (
                    best_val_metric, val_metric))
                best_val_metric = val_metric
                best_model_path = model_path
                best_param = param

            search_res_list.append(run_dict)
            print(str(run_dict))

            if task_type == 'reg':
                save_grid_search_res(
                    '../model_search_res/%s/mlp_%s_%s.csv' % (data_version, train_mode, task_type),
                    search_res_list,
                    columns=['beta', 'init_learning_rate', 'bs', 'feature_dropout', 'repeat', 'rmse',
                             'map@10', 'ndcg@10', 'map@30', 'ndcg@30', 'map@50', 'ndcg@50']
                )
            else:
                save_grid_search_res(
                    '../model_search_res/%s/mlp_%s_%s.csv' % (data_version, train_mode, task_type),
                    search_res_list,
                    columns=['pos_weight', 'beta', 'init_learning_rate', 'bs', 'feature_dropout',
                             'repeat', 'precision', 'recall', 'acc'], sort_col='precision'
                )

    if best_model_path is not None:
        model = DaCityModel(input_dim=x_target.shape[-1], task_type=task_type)
        rank_metrics = evaluate(model, Solver, best_model_path, x_target, y_target)
        print('Best parameter: ' + str(best_param))
        print('Best metric:' + str(rank_metrics))


if __name__ == '__main__':
    run_mlp_one_time(
        train_mode='source', task_type='reg', city_pair=('sh', 'nb'),
        path_pattern='../data/road/train_bound/%s_500_week.csv',
        hot_count=100
    )
    run_mlp_one_time(
        train_mode='dann', task_type='reg', city_pair=('sh', 'nb'),
        path_pattern='../data/road/train_bound/%s_500_week.csv',
        hot_count=100
    )

    grid_search_mlp(train_mode='source', search_mode='grid', repeat=1, city_pair=('sh', 'nb'),
                    data_version='bound', task_type='reg', hot_count=100)
    grid_search_mlp(train_mode='dann', search_mode='grid', repeat=1, city_pair=('sh', 'nb'),
                    data_version='bound', task_type='reg', hot_count=100)
