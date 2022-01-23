import numpy as np
from clib import run, extract_seizure_event, epileptor
from utils import loadConnection
from multiprocessing import Pool
import itertools
from scipy.special import comb, perm

def propagation_FIP(x_0s, conn, D_1=0.025, D_2=0.25):
    def setParameters(network):
        for i in range(network.n):
            for j in range(network.n):
                network.K[i][j] = conn[i][j]
        for i in range(network.n):
            network.vars[i].contents.paras.contents.x_0 = x_0s[i]
            network.vars[i].contents.paras.contents.D_1 = D_1
            network.vars[i].contents.paras.contents.D_2 = D_2
    return setParameters

def run_propagation(x_0s, conn, duration=3000, D_1=0.025, D_2=0.25):
    dt = 0.05
    ic = int(duration / dt)
    result, network = run(dt, ic, propagation_FIP(x_0s, conn, D_1=D_1, D_2=D_2), stable=True)
    ses = extract_seizure_event(result, network.n, ic)
    epileptor.free_data(result, network)
    return ses

class Patient():
    
    def __init__(self, SC, ex):
        self.SC = SC # 结构连接
        self.ex = ex # 兴奋性
        self.EZ = np.argmax(ex) # 致痫区
        self.n = len(SC) # 脑区数
    
    # 发作,运用癫痫子模型模拟癫痫发作结果
    # 返回各脑区发作事件
    def seizure(self, duration=3000, D_1=0.025, D_2=0.25):
        return run_propagation(self.ex, self.SC, duration, D_1=D_1, D_2=D_2)

    # 判断病人是否是seizure free的(即只有1个脑区(EZ)有癫痫发作)
    # def is_seizure_free(self, try_count=30, duration=3000, D_1=0.025, D_2=0.25):
    #     count = 0
    #     while (count < try_count):
    #         ses = self.seizure(duration=duration, D_1=D_1, D_2=D_2)
    #         for i in range(self.EZ):
    #             if len(ses[i]) > 0:
    #                 return False
    #         for i in range(self.EZ + 1, len(ses)):
    #             if len(ses[i]) > 0:
    #                 return False
    #         count += 1
    #     return True

    # 检查是否存在其他致痫区
    def check_seizure_no_spread(self):
        ses = run_propagation(self.ex, self.SC)
        t_onset = np.array(list(map(lambda se: se[0] if len(se) > 0 else -1, ses)))
        onset_sorted = np.argsort(t_onset)
        onset_sorted = onset_sorted[np.where(t_onset[onset_sorted] < 0)[0]]
        if len(onset_sorted) <= 1:
            return []
        EZ = onset_sorted[0]
        on_group = [EZ]
        result_group = []
        for node in onset_sorted[1:]:
            if np.all(self.SC[on_group, node] == 0):
                result_group.append(node)
            on_group.append(node)
        return result_group

    # 计算关键节点
    def find_key_nodes(self):
        nodes = []
        SC = np.copy(self.SC)
        while len(nodes) < self.n - 1:
            ses = run_propagation(self.ex, SC, duration=3000, D_1=0, D_2=0)
            t_onset = np.array(list(map(lambda se: se[0] if len(se) > 0 else -1, ses)))
            onset_sorted = np.argsort(t_onset)
            onset_sorted = onset_sorted[np.where(t_onset[onset_sorted] > 0)[0]]
            if len(onset_sorted) <= 1:
                break
            node = onset_sorted[1]
            nodes.append(node)
            SC[self.EZ, node] = 0
            SC[node, self.EZ] = 0
        return nodes

    # 模拟切除后的活动
    def run_after_cut(self, cut, try_count=30, duration=3000, D_1=0.025, D_2=0.25):
        SC = np.copy(self.SC)
        SC[self.EZ, cut] = 0
        SC[cut, self.EZ] = 0
        sess = np.empty(try_count, object)
        for i in range(try_count):
            ses = run_propagation(self.ex, SC, duration, D_1=D_1, D_2=D_2)
            sess[i] = ses
        return sess

    def run_after_lower_ex(self, lower_nodes, lower_values, try_count=30, duration=3000, D_1=0.025, D_2=0.25):
        ex = np.copy(self.ex)
        ex[lower_nodes] += lower_values
        sess = np.empty(try_count, object)
        for i in range(try_count):
            ses = run_propagation(ex, self.SC, duration, D_1=D_1, D_2=D_2)
            sess[i] = ses
        return sess

    def is_seizure_free_from_sess(self, sess):
        for ses in sess:
            for i in range(self.EZ):
                if len(ses[i]) > 0:
                    return False
            for i in range(self.EZ + 1, len(ses)):
                if len(ses[i]) > 0:
                    return False
        return True

def control_ex_old(n, EZ=None):
    x_0s = np.full((n), -2.12)
    if EZ is None:
        x_0s[np.random.randint(n)] = - 2.1 + np.random.rand(1) * 0.1 + 1.1
    else:
        x_0s[EZ] = - 2.1 + np.random.rand(1) * 0.1 + 1.1
    return x_0s

def control_ex(n, EZ=None):
    x_0s = np.full((n), np.random.uniform(-2.16, -2.08))
    if EZ is None:
        x_0s[np.random.randint(n)] = - 2.1 + np.random.rand(1) * 0.1 + 1.1
    else:
        x_0s[EZ] = - 2.1 + np.random.rand(1) * 0.1 + 1.1
    return x_0s

def exp_ex(n, EZ=None):
    x_0s = - 2.1 + np.random.standard_exponential(n) * 0.02 - 0.012
    while np.any(x_0s > - 2.06):
        x_0s = - 2.1 + np.random.standard_exponential(n) * 0.02 - 0.012
    if EZ is None:
        x_0s[np.random.randint(n)] = - 2.1 + np.random.rand(1) * 0.1 + 1.1
    else:
        x_0s[EZ] = - 2.1 + np.random.rand(1) * 0.1 + 1.1
    return x_0s

def norm_ex(n, EZ=None):
    x_0s = np.random.normal(-2.12, 0.03, n)
    while np.any(x_0s > - 2.06):
        x_0s = np.random.normal(-2.12, 0.03, n)
    if EZ is None:
        x_0s[np.random.randint(n)] = - 2.1 + np.random.rand(1) * 0.1 + 1.1
    else:
        x_0s[EZ] = - 2.1 + np.random.rand(1) * 0.1 + 1.1
    return x_0s

def uniform_ex(n, EZ=None):
    x_0s = np.random.uniform(-2.16, -2.08, n)
    if EZ is None:
        x_0s[np.random.randint(n)] = - 2.1 + np.random.rand(1) * 0.1 + 1.1
    else:
        x_0s[EZ] = - 2.1 + np.random.rand(1) * 0.1 + 1.1
    return x_0s

def normal_ex(n, min_loc=-2.14, max_loc=-2.1, scale=0.03, EZ=None):
    def gen():
        return np.random.normal(np.random.uniform(min_loc, max_loc), scale, n)
    x_0s = gen()
    while np.any(x_0s > - 2.06):
        x_0s = gen()
    if EZ is None:
        EZ = np.random.randint(n)
    x_0s[EZ] = np.random.uniform(-1, -0.9)
    return x_0s

#================无噪声==================

# 对每个致痫区计算并保持控制组发作情况
def run_control_patients():
    SC = loadConnection()
    n = len(SC)
    sess = np.empty((n), object)
    for i in range(n):
        ex = control_ex_old(n, EZ=i)
        # p = Patient(SC, ex)
        # # sess[i] = p.seizure(D_1=0, D_2=0)
        sess[i] = run_propagation(ex, SC, D_1=0, D_2=0)
    np.save('results/results_control_patients.npy', sess)

# 使用延迟改变来量化兴奋性对传播的影响
# def delay_effect(id):
    # result = np.load('results/results_patients_{}.npz'.format(id), allow_pickle=True)
    # patients = result['patients']
    # sess = result['sess']
    # control_sess = np.load('results/results_control_patients.npy')
    # pass

# 使用发作顺序改变来量化兴奋性对传播的影响
# def order_effect(id):
    # result = np.load('results/results_patients_{}.npz'.format(id), allow_pickle=True)
    # patients = result['patients']
    # sess = result['sess']
    # control_sess = np.load('results/results_control_patients.npy')
    # pass

def sub_run_patients(patient):
    return patient.seizure(D_1=0, D_2=0), patient.find_key_nodes()

# 批量生成病人并保存发作情况与关键节点
def run_patients(id, count, ex_fun, ex_kw={}, g_conn=1):
    SC = loadConnection() * g_conn
    n = len(SC)
    patients = np.empty((count), object)
    sess = np.empty((count), object)
    key_nodes = np.empty((count), object)

    def callback_fun(i):
        def callback(result):
            sess[i] = result[0]
            key_nodes[i] = result[1]
        return callback

    pool = Pool()
    for i in range(count):
        ex = ex_fun(n, **ex_kw)
        patients[i] = Patient(SC, ex)
        pool.apply_async(sub_run_patients, (patients[i],), callback=callback_fun(i), error_callback=print)
    pool.close()
    pool.join()

    np.savez('results/results_patients_{}.npz'.format(id), patients=patients, sess=sess, key_nodes=key_nodes)

# 生成集合(一维数组)的全部子集
def all_sub_set(K, include_empty=False, include_full=True, keep_order=True):
    n = len(K)
    # 避免结果过多导致内存溢出等问题
    if n > (23 if keep_order else 10):
        raise OverflowError("length of K is too large")
    all_subKs = []
    if include_empty:
        all_subKs.append(tuple())
    fun = itertools.combinations if keep_order else itertools.permutations
    for i in range(1, n):
        all_subKs.extend(list(fun(K, i)))
    if include_full:
        all_subKs.extend(list(fun(K, n)))
    return all_subKs

# 生成集合(一维数组)的任意子集(组合等概率)
def random_sub_set(K, count=1, include_empty=False, include_full=True, keep_order=True):
    n = len(K)
    if n == 0:
        return []
    subKs = []
    for _ in range(count):
        while True:
            f = np.random.uniform(size=(n)) > 0.5
            subK = np.array(K)[f]
            if not include_empty and len(subK) == 0:
                continue
            if not include_full and len(subK) == n:
                continue
            break
        if not keep_order:
            np.random.shuffle(subK)
        subKs.append(tuple(subK))
    return subKs

# 生成集合(一维数组)的任意子集(排列等概率)
def random_sub_set_perm(K, count=1, include_empty=False, include_full=True):
    n = len(K)
    if n == 0:
        return []
    ms = range(0 if include_empty else 1, n+1 if include_full else n)
    p = list(map(lambda x: perm(n, x), ms))
    p = np.cumsum(p)
    p /= p[-1]
    subKs = []
    for _ in range(count):
        m = ms[len(np.where(np.random.uniform()>p)[0])]
        subK = np.random.choice(K, m, False)
        subKs.append(tuple(subK))
    return subKs

# 采样一个有关键节点的病人
def choice_patient_by_key_nodes_count(id, count=1, min_knc=1, max_knc=None, replace=True):
    # 读取病人集合
    results = np.load('results/results_patients_{}.npz'.format(id), allow_pickle=True)
    patients = results['patients']
    key_nodes = results['key_nodes']

    knc = np.array(list(map(lambda x: len(x), key_nodes)))
    if max_knc is None:
        max_knc = np.max(knc)
    filtered = np.where(np.all([knc >= min_knc, knc <= max_knc], axis=0))[0]
    choiced = np.random.choice(filtered, count, replace=replace)
    return patients[choiced], key_nodes[choiced]

# 为了控制癫痫传播(无症状癫痫),必须将所有关键节点切除.
# 证明:任意取R为K的真子集,都无法完全控制癫痫传播,当R=K时能完全控制癫痫传播
# 验证上述观点
# job_id 任务id
# set_id 用来采样的病人集合
# count 采样的病人数
# patient_filter 病人采样函数, 采样函数应当从集合中返回一个病人的子集和关键节点的子集
# pf_kw 传递到patient_filter的关键字参数
# sub_set_filter 子集采样函数, 采样函数应当返回一个元素为元组的list, 默认使用等组合概率随机单次采样
def run_sub_key_nodes_cut(job_id, set_id, count, patient_filter=choice_patient_by_key_nodes_count, pf_kw={}, sub_set_filter=random_sub_set, ssf_kw={}):

    results = []
    # 采样病人
    patients, key_nodes = patient_filter(set_id, count, **pf_kw)
    pool = Pool()
    for p, kns in zip(patients, key_nodes):
        # 采样关键节点子集
        skns = sub_set_filter(kns, **ssf_kw)
        # patient_result = dict(patient=p, results=[])
        patient_result = dict(patient=p, key_nodes=kns, results=[]) # change after job 1
        results.append(patient_result)
        for j in range(len(skns)):
            cut = list(skns[j])
            cut_result = dict(cut=cut, results=[])
            patient_result['results'].append(cut_result)
            pool.apply_async(Patient.run_after_cut, (p, cut), dict(try_count=1, D_1=0, D_2=0), callback=cut_result['results'].extend)
    pool.close()
    pool.join()
    np.save('results/results_sub_key_nodes_cut_{}.npy'.format(job_id), results)

# patient 病人, lower_nodes 要降低的节点索引（列表）, lower_scale 节点降低的系数（默认为1,会被归一化到均值为1）
# return 使病人无发作的平均节点兴奋度降低值， 二分法具体过程
def simulated_lower_ex(patient, lower_nodes, lower_scale=1, error=1e-4, le_kw=dict(try_count=1, D_1=0, D_2=0)):
    lower_scale /= np.mean(lower_scale)
    process = []
    min_lower_values = 0
    max_lower_values = - 2
    sess = patient.run_after_lower_ex(lower_nodes, lower_scale * max_lower_values, **le_kw)
    process.append({'lower_values': lower_scale * max_lower_values, 'sess': sess})
    if not patient.is_seizure_free_from_sess(sess):
        return - np.inf, process
    while abs(min_lower_values - max_lower_values) > error:
        lower_values = (min_lower_values + max_lower_values) / 2
        sess = patient.run_after_lower_ex(lower_nodes, lower_scale * lower_values, **le_kw)
        process.append({'lower_values': lower_scale * lower_values, 'sess': sess})
        if patient.is_seizure_free_from_sess(sess):
            max_lower_values = lower_values
        else:
            min_lower_values = lower_values
    return max_lower_values, process

# def sub_run_lower_ex(p, kns, sle_kw):
#     nodes = np.arange(p.n)
#     nodes = np.delete(nodes, p.EZ)
#     values_all, process_all = simulated_lower_ex(p, nodes, **sle_kw)
#     values_key, process_key = simulated_lower_ex(p, kns, **sle_kw)
#     key_rwer = rwer2(p)
#     values_rwer, process_rwer = simulated_lower_ex(p, kns, lower_scale=key_rwer[kns], **sle_kw)
#     kns_rwer = np.where(key_rwer > 0.0126)[0]
#     values_rwer_key, process_rwer_key = simulated_lower_ex(p, kns_rwer, lower_scale=key_rwer[kns_rwer], **sle_kw)
#     return dict(patient=p, key_nodes=kns, kns_rwer=kns_rwer, values_all=values_all, process_all=process_all, values_key=values_key, process_key = process_key, 
#         values_rwer=values_rwer, process_rwer=process_rwer, values_rwer_key=values_rwer_key, process_rwer_key=process_rwer_key)

def sub_run_lower_ex(p, kns, sle_kw):
    nodes = np.arange(p.n)
    nodes = np.delete(nodes, p.EZ)
    values_all, process_all = simulated_lower_ex(p, nodes, **sle_kw)
    values_key, process_key = simulated_lower_ex(p, kns, **sle_kw)
    key_strength = p.SC[p.EZ]
    values_strength, process_strength = simulated_lower_ex(p, kns, lower_scale=key_strength[kns], **sle_kw)
    kns_strength = np.where(key_strength > 0.02259)[0]
    values_strength_key, process_strength_key = simulated_lower_ex(p, kns_strength, lower_scale=key_strength[kns_strength], **sle_kw)
    return dict(patient=p, key_nodes=kns, kns_strength=kns_strength, values_all=values_all, process_all=process_all, values_key=values_key, process_key = process_key, 
        values_strength=values_strength, process_strength=process_strength, values_strength_key=values_strength_key, process_strength_key=process_strength_key)


def run_lower_ex(job_id, set_id, count, patient_filter=choice_patient_by_key_nodes_count, pf_kw={}, sle_kw={}):
    results = []
    # 采样病人
    patients, key_nodes = patient_filter(set_id, count, **pf_kw)
    pool = Pool()
    for p, kns in zip(patients, key_nodes):
        pool.apply_async(sub_run_lower_ex, (p, kns, sle_kw), callback=results.append)
    pool.close()
    pool.join()
    np.save('results/results_lower_ex_{}.npy'.format(job_id), results)

# 调整阈值改变关键节点数量，在不同的阈值下计算损伤率和控制率，记录阈值，关键节点和切除后发作
# least_keep: 最少保留比例，0表示完全不保留， 必须大于等于0
# least_cut: 最少切除比例，0表示完全切除, 1表示完全不切除， 必须小于等于1
# least_cut必须大于least_keep
def run_threshold_cut(job_id, set_id, count, key_fun, curve_dots=10, least_keep=0, least_cut=1, patient_filter=choice_patient_by_key_nodes_count, pf_kw={}):
    if (least_keep > least_cut):
        raise "least_keep is larger than least_cut"
    if (least_cut > 1):
        least_cut = 1
    if (least_keep < 0):
        least_keep = 0
    results = []
    patients, key_nodes = patient_filter(set_id, count, **pf_kw)
    key_values = list(map(key_fun, patients))
    # convert key nodes
    pred = np.array([])
    # real = np.array([])
    # for p, k in zip(patients, key_nodes):
    #     y = np.zeros((len(p.SC)))
    #     y[k] = 1
    #     real = np.concatenate([real, y])
    for kv in key_values:
        pred = np.concatenate([pred, kv])
    sorted_pred = np.sort(pred)
    l = len(sorted_pred)
    ti = np.linspace(int(least_keep * l), int(least_cut * l) - 1, curve_dots, dtype=int)
    # thresholds = (sorted_pred[ti] + sorted_pred[ti + 1]) / 2
    thresholds = sorted_pred[ti]
    # 确保完全不切除
    if (least_cut >= 1):
        thresholds[-1] = np.inf
    pool = Pool()
    for p, kns in zip(patients, key_nodes):
        keys = key_fun(p)
        p_result = dict(patient=p, key_nodes=kns, keys=keys, results=[])
        results.append(p_result)
        for t in thresholds:
            cut = np.where(keys >= t)[0]
            t_result = dict(threshold=t, cut=cut, results=[])
            p_result['results'].append(t_result)
            pool.apply_async(Patient.run_after_cut, (p, cut), dict(try_count=1, D_1=0, D_2=0), callback=t_result['results'].extend)
    pool.close()
    pool.join()
    np.save('results/results_threshold_cut_{}.npy'.format(job_id), results)

#================无噪声==================

#===RWER====

def x_0s_revise_matrix(conn, x_0s):
    n = conn.shape[0]
    return x_0s - 0.1 * x_0s * np.matmul(np.ones((1, n)), conn).flatten() + 0.1 * np.matmul(conn, np.reshape(x_0s, (n, 1))).flatten()

def RWER2_matrix(RNA, c, q):
    n = RNA.shape[0]
    I = np.identity(n)
    C = np.diag(c)
    B = np.matmul(RNA, I - C).T + \
        np.matmul(np.reshape(q, (n, 1)), np.transpose(np.matmul(RNA, np.reshape(c, (n, 1))) - 1))
    return np.matmul(np.linalg.inv(I - B), q)

# \bold(A) is the adjacency matrix
# return: row-normalized matrix
def RWER_A2RNA(A):
    d = np.sum(A, axis=1)
    d[d==0] = 1
    ID = np.diag(1. / d)
    return np.matmul(ID, A)

def rwer2(patient, b=22, p=0):
    SC = patient.SC
    exs = patient.ex
    n = patient.n
    EZ = patient.EZ
    A = RWER_A2RNA(SC)

    D = np.reshape(np.sum(SC, axis=0), (n,1))
    f = (1 - p) * (D / np.max(D))
    # f = np.full((n,1), 1 - p)
    EZ = np.argmax(exs)
    f[EZ] = 1
    A = f * A
    A[np.arange(n), np.arange(n)] =  1 - f.flatten()

    c = (1. / (1 + np.exp((x_0s_revise_matrix(SC, exs) - (-2.05)) * b)))
    # c[EZ] = 0
    q = np.zeros((n))
    q[EZ] = 1
    rp = RWER2_matrix(A, c, q)
    rp *= np.sum(SC[EZ])
    # rp *= (exs[EZ] + 2)
    rp[EZ] = 0
    return rp

#===========

if __name__== "__main__":
    # run_control_patients()

    # run_patients(0, 400, control_ex)

    # run_patients(1, 400, exp_ex)
    
    # run_patients(2, 400, norm_ex)

    # run_patients(3, 400, uniform_ex)

    # 抽样6个病人全子集(打乱顺序)
    # run_sub_key_nodes_cut(0, 2, 6, pf_kw=dict(min_knc=2, max_knc=5, replace=False), sub_set_filter=all_sub_set, ssf_kw=dict(keep_order=False))

    # 从病人集合2抽样3000组病人子集(打乱顺序)
    # run_sub_key_nodes_cut(1, 2, 3000, sub_set_filter=random_sub_set_perm)

    # run_patients(4, 1200, normal_ex)

    # run_patients(21, 400, normal_ex, dict(min_loc=-2.12,max_loc=-2.12,scale=0.02))
    # run_patients(22, 400, normal_ex, dict(min_loc=-2.12,max_loc=-2.12,scale=0.04))
    # run_patients(23, 400, normal_ex, dict(min_loc=-2.14,max_loc=-2.14))
    # run_patients(24, 400, normal_ex, dict(min_loc=-2.13,max_loc=-2.13))
    # run_patients(25, 400, normal_ex, dict(min_loc=-2.11,max_loc=-2.11))
    # run_patients(26, 400, normal_ex, dict(min_loc=-2.1,max_loc=-2.1))
    # for i, v in enumerate(np.linspace(0, 0.04, 9)):
    #     run_patients(210+i, 800, normal_ex, dict(min_loc=-2.12, max_loc=-2.12, scale=v))
    # for i, v in enumerate(np.linspace(-2.15, -2.11, 9)):
    #     run_patients(220+i, 800, normal_ex, dict(min_loc=v, max_loc=v, scale=0.04))

    # run_patients('00', 1200, control_ex)
    # run_patients(10, 1200, exp_ex)
    # run_patients(30, 1200, uniform_ex)
    # run_lower_ex(1, 4, 400, pf_kw=dict(replace=False))
    # run_lower_ex(2, 4, 400, pf_kw=dict(replace=False))
    # run_lower_ex(3, 4, 400, pf_kw=dict(replace=False))

    # 这两个例子是修改前跑的测试例子，不包括边界点的数据
    # run_threshold_cut(1, 4, 400, lambda p: p.SC[p.EZ], curve_dots=20, pf_kw=dict(replace=False))
    # run_threshold_cut(2, 4, 400, rwer2, curve_dots=20, pf_kw=dict(replace=False))
    
    # run_threshold_cut(3, 4, 400, lambda p: p.SC[p.EZ], curve_dots=20, least_keep=0.8, pf_kw=dict(replace=False))
    # run_threshold_cut(4, 4, 400, rwer2, curve_dots=20, least_keep=0.8, pf_kw=dict(replace=False))

    # run_threshold_cut(0, 4, 400, lambda p: np.random.uniform(0, 1, (len(p.SC))), curve_dots=20, pf_kw=dict(replace=False))
    # run_threshold_cut(5, 4, 400, lambda p: np.random.uniform(0, 1, (len(p.SC))), curve_dots=20, least_keep=0.8, pf_kw=dict(replace=False))

    # 多倍连接强度，更低平均兴奋性，更大兴奋性方差
    # for i, v in enumerate(np.linspace(-2.2, -2.16, 9)):
    #     run_patients(910+i, 800, normal_ex, dict(min_loc=v, max_loc=v, scale=0.06), 2)

    # for i, v in enumerate(np.linspace(0.04, 0.08, 9)):
    #     run_patients(920+i, 800, normal_ex, dict(min_loc=-2.18, max_loc=-2.18, scale=v), 2)

    # 高平均同质兴奋性的对照组
    # run_patients('01', 800, normal_ex, dict(min_loc=-2.1, max_loc=-2.1, scale=0))
    # 低平均同质兴奋性的对照组
    # run_patients('02', 800, normal_ex, dict(min_loc=-2.15, max_loc=-2.15, scale=0))

    # run_threshold_cut(6, '01', 400, lambda p: p.SC[p.EZ], curve_dots=20, least_keep=0.8, pf_kw=dict(replace=False))
    # run_threshold_cut(7, '00', 400, lambda p: p.SC[p.EZ], curve_dots=20, least_keep=0.8, pf_kw=dict(replace=False))

    # 多倍连接强度，更低平均兴奋性，更大兴奋性方差
    # for i, v in enumerate(np.linspace(-2.22, -2.18, 9)):
    #     run_patients(930+i, 800, normal_ex, dict(min_loc=v, max_loc=v, scale=0.08), 3)

    # for i, v in enumerate(np.linspace(0.06, 0.1, 9)):
    #     run_patients(940+i, 800, normal_ex, dict(min_loc=-2.2, max_loc=-2.2, scale=v), 3)

    # run_threshold_cut(10, 210, 400, lambda p: p.SC[p.EZ], curve_dots=20, pf_kw=dict(replace=False))
    # run_threshold_cut(11, 210, 400, lambda p: np.random.uniform(0, 1, (len(p.SC))), curve_dots=20, pf_kw=dict(replace=False))

    # run_threshold_cut(12, 210, 400, lambda p: p.SC[p.EZ], curve_dots=20, least_keep=0.8, pf_kw=dict(replace=False))
    # for i in range(1, 9):
    #     run_threshold_cut(12+i, 210+i, 400, lambda p: p.SC[p.EZ], curve_dots=20, least_keep=0.8, pf_kw=dict(replace=False))

    # run_threshold_cut(22, 210, 400, rwer2, curve_dots=20, least_keep=0.8, pf_kw=dict(replace=False))
    # for i in range(1, 9):
    #     run_threshold_cut(22+i, 210+i, 400, rwer2, curve_dots=20, least_keep=0.8, pf_kw=dict(replace=False))


    # 补充计算
    for i, v in enumerate(np.linspace(0, 0.04, 5)):
        run_patients(2210+i, 3200, normal_ex, dict(min_loc=-2.12, max_loc=-2.12, scale=v))
