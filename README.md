# misc

# tensorflow demo
https://www.tuicool.com/articles/vyiYJzu
https://github.com/tobegit3hub/tensorflow_template_application

class PointsCompression(object):
  def __init__(self):
	  self.table = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-'

  def parseToPoints(self, s):
    points = s.split(",")
    result = []
    for p in points:
        coord = p.split(" ")
        result.append([float(coord[1]), float(coord[0])])
    return result

  def evaluate(self, s, percision=100000):
    points = self.parseToPoints(s)
    lat, lon = 0, 0
    result = []
    for point in points:
      newLat = int(round(point[0] * percision))
      newLon = int(round(point[1] * percision))
      dx = newLat - lat
      dy = newLon - lon
      record = str([newLat, newLon])+' [ '+str(dx)+', '+str(dy)
      lat = newLat
      lon = newLon
      # 循环左移: *2, 负数转正数-1
      dy = (dy << 1) ^ (dy >> 31)
      dx = (dx << 1) ^ (dx >> 31)
      index = (dx + dy) * (dx + dy + 1) / 2 + dx
      record = record +'] [' + str(index) + ' ]'
      while index > 0:
        rem = index & 31
        index = (index - rem) / 32
        if index > 0:
            rem = rem + 32
        result.append(self.table[rem])
    return "".join(result)

const table = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-';
function decompressPoints(s) {
  const array = s.split('');
  const nums = [];
  let num = 0;
  let i = array.length - 1;
  for (; i >= 1; i -= 1) {
    const current = table.indexOf(array[i]);
    const previous = table.indexOf(array[i - 1]);
    if (current < 32) {
      num = current * 32;
    } else {
      num += current - 32;
      if (previous < 32) {
        nums.push(num);
      } else {
        num *= 32;
      }
    }
  }
  nums.push(num);
  const current = table.indexOf(array[i]);
  nums[nums.length - 1] = nums[nums.length - 1] + (current - 32);
  const coords = nums.map((e) => {
    let x = Math.sqrt(e * 2);
    x = Math.floor(x);
    let lat = e - x * (x + 1) / 2;
    while (lat < 0) {
      x -= 1;
      lat = e - x * (x + 1) / 2;
    }
    let lon = x - lat
    if (lat % 2 === 0) {
      lat /= 2;
    } else {
      lat = -((lat + 1) / 2);
    }
    if (lon % 2 === 0) {
      lon /= 2;
    } else {
      lon = -((lon + 1) / 2);
    }
    return [lat, lon];
  })
  for (let j = coords.length - 2; j >= 0; j -= 1) {
    const latLon = coords[j];
    latLon[0] += coords[j + 1][0];
    latLon[1] += coords[j + 1][1];
  }
  const result = coords.map((e) => {
    // return [e[1] / 100000.0, e[0] / 100000.0];
    return new window.AMap.LngLat(e[1] / 100000.0, e[0] / 100000.0);
  });
  return result;
}

def DefaultValueForZero(fval, default_value, kMissingValueRange = 1e-20):
    if fval > -kMissingValueRange and fval <= kMissingValueRange:
        return default_value
    else:
        return fval

def predict(bst, vec):
    dump_model = bst.dump_model()

    def dfs(root, node_path, node_value):
        if "split_feature" in root:
            node_path.append(root['split_feature'])
            node_value.append(root['internal_value'])
            split_idx = root['split_feature']

            default_left = root['default_left']

            goLeft, goRight = False, False
            val = vec[split_idx]

            if val < -1e-35: # 特征缺失, < 0 表示特征缺失
                if default_left:
                    return dfs(root["left_child"], node_path, node_value)
                else:
                    return dfs(root["right_child"], node_path, node_value)
            else:
                decision_type = root['decision_type']
                if root['decision_type'] == '==':
                    val = int(val)
                    if str(val) in root['threshold'].split('||'):
                        return dfs(root["left_child"], node_path, node_value)
                    else:
                        return dfs(root["right_child"], node_path, node_value)
                elif root['decision_type'] == '<=':
                    if val > root['threshold']:
                        return dfs(root["right_child"], node_path, node_value)
                    else:
                        return dfs(root["left_child"], node_path, node_value)
                else:
                    raise Exception('UnKnown decision_type %s' %root['decision_type'])
        else: # leaf node
            node_value.append(root['leaf_value'])
            return root['leaf_index'], root['leaf_value']

    num_class = dump_model['num_class']
    vvv = [0.0] * num_class

    feature_gain_dict = collections.defaultdict(float)
    for tree in dump_model["tree_info"]:
        # print tree['tree_index'], tree['shrinkage']
        node_path = []
        node_value = []
        leaf_index, leaf_value = dfs(tree["tree_structure"], node_path, node_value) #, node_path, node_value

        node_value[-1] = node_value[-1] / tree['shrinkage'] # 叶子节点增益还原

        for i in range(0, len(node_path)):
            feat_idx = node_path[i]
            gain = node_value[i+1] - node_value[i]
            feature_gain_dict[feat_idx] += gain

        vvv[tree['tree_index'] % num_class] += leaf_value # * tree['shrinkage']

    print feature_gain_dict.items(), sum(feature_gain_dict.values())
    return vvv

def sigmoid(v):
  return 1.0 / (1 + math.exp(-v))

if __name__ == '__main__':
  modelName, trainFileName, testFileName = sys.argv[1], sys.argv[2], sys.argv[3]
  gbm = lgb.Booster(model_file=modelName)

  X_train, Y_train = load_svmlight_file(trainFileName, zero_based=True)
  X_test, Y_test = load_svmlight_file(testFileName, zero_based=True)

  # P_test = gbm.predict(X_test, pred_leaf=True)
  P_test = gbm.predict(X_test)
  # print P_test[0: 10]

  # print Y_test.shape, P_test.shape
  labels, orig_predictions = Y_test, P_test
  auc = sklearn.metrics.roc_auc_score(y_true=labels, y_score=orig_predictions)
  mae = sklearn.metrics.mean_absolute_error(y_true=labels, y_pred=orig_predictions)
  logloss = sklearn.metrics.log_loss(y_true=labels, y_pred=orig_predictions)

  print auc, logloss, mae

  # 0.79321950613573444 1 1:5.7 2:2.8 3:4.1 4:1.3

  with open('tree.info', 'w') as fp:
    json.dump(gbm.dump_model(), fp)

  mock_vec = [0, 5.7, 2.8, 4.1, 1.3]
  score = predict(gbm, mock_vec)
  print 'score:', score, 'prob:', sigmoid(score[0])

  # from lime.lime_text import LimeTextExplainer
  # explainer = LimeTextExplainer(class_names=['c0', 'c1'])
  import lime.lime_tabular
  # explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=['f1', 'f2', 'f3', 'f4'], class_names=['c0', 'c1'], training_labels=Y_train)
  # 'quartile', 'decile', 'entropy'
  # explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=['f1', 'f2', 'f3', 'f4'], class_names=['c0', 'c1'], discretize_continuous=False, training_labels=Y_train)
  explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=['-', 'f1', 'f2', 'f3', 'f4'], class_names=['c0', 'c1'], discretize_continuous=False)
  # explainer = lime.lime_tabular.LimeTabularExplainer(X_train, class_names=['c0', 'c1'])

  # exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)
  def limeExpProbFunc(vec):
    score = predict(gbm, vec)
    prob = sigmoid(score[0])
    return np.array(1 - prob, prob)

  print 'gbm predict:', gbm.predict(np.array([mock_vec]))
  def limeExpProbFunc2(x):
    prob = gbm.predict(x)
    # prob = prob.tolist()

    ret = np.array([1 - prob, prob]).T
    # print ret
    return ret

  # print limeExpProbFunc2(mock_vec)
  exp = explainer.explain_instance(np.array(mock_vec), limeExpProbFunc2, num_features=5)
  print mock_vec
  print exp.as_list()
