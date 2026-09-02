<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import InteractionGraph from './components/InteractionGraph.vue'
import RiskChart from './components/RiskChart.vue'
import ResultPanel from './components/ResultPanel.vue'

const view = ref('dashboard')
const loading = ref(false)
const error = ref('')
const metrics = ref(null)
const weiboResult = ref(null)
const weiboForm = reactive({ target: '', recent_posts: 20 })

const cards = computed(() => {
  const profile = metrics.value?.misbot_profile?.user_profile
  const account = (metrics.value?.multiagent_advanced_metrics ?? metrics.value?.multiagent_baseline_metrics)?.decision_agent
  const information = (metrics.value?.information_advanced_metrics ?? metrics.value?.information_agent_metrics)?.decision_agent
  return [
    ['活跃标注用户', profile?.records?.toLocaleString() ?? '—'],
    ['标注机器人', profile?.labels?.['1']?.toLocaleString() ?? '—'],
    ['账号检测 AUC', account?.roc_auc?.toFixed(3) ?? '—'],
    ['信息检测 AUC', information?.roc_auc?.toFixed(3) ?? '—'],
  ]
})
const accountChart = computed(() => chartRows(metrics.value?.multiagent_advanced_metrics ?? metrics.value?.multiagent_baseline_metrics, ['behavior_agent', 'text_agent', 'decision_agent']))
const informationChart = computed(() => chartRows(metrics.value?.information_advanced_metrics ?? metrics.value?.information_agent_metrics, ['relation_agent', 'propagation_agent', 'decision_agent']))

async function request(url, options) {
  loading.value = true
  error.value = ''
  try {
    const response = await fetch(url, options)
    const body = await response.json().catch(() => null)
    if (!response.ok) throw new Error(body?.detail || `请求失败：${response.status}`)
    return body
  } catch (reason) {
    error.value = reason.message
    throw reason
  } finally {
    loading.value = false
  }
}

async function detectWeibo() {
  weiboResult.value = await request('/api/detect/weibo', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(weiboForm),
  })
}

function chartRows(source, names) {
  const labels = { behavior_agent: '行为', text_agent: '文本', relation_agent: '关系', propagation_agent: '传播', decision_agent: '融合' }
  return names.map(name => ({ name: labels[name], value: source?.[name]?.roc_auc ?? 0 }))
}

onMounted(async () => { metrics.value = await request('/api/metrics') })
</script>

<template>
  <div class="shell">
    <aside>
      <div class="brand"><span class="brand-mark">B</span><div><strong>BotArena</strong><small>多智能体检测平台</small></div></div>
      <nav aria-label="主导航">
        <button :class="{ active: view === 'dashboard' }" @click="view = 'dashboard'">态势总览</button>
        <button :class="{ active: view === 'weibo' }" @click="view = 'weibo'">微博协同检测</button>
      </nav>
      <div class="status"><i />模型服务已连接</div>
    </aside>

    <main>
      <header><div><p>微博社交机器人对抗分析</p><h1>{{ view === 'dashboard' ? '态势总览' : '微博账号与信息协同检测' }}</h1></div><span class="tag">异构四智能体协同</span></header>
      <p v-if="error" class="error">{{ error }}</p>

      <template v-if="view === 'dashboard'">
        <section class="cards"><article v-for="card in cards" :key="card[0]"><span>{{ card[0] }}</span><strong>{{ card[1] }}</strong></article></section>
        <section class="grid two"><article class="panel"><RiskChart title="账号智能体 ROC-AUC" :rows="accountChart" /></article><article class="panel"><RiskChart title="信息智能体 ROC-AUC" :rows="informationChart" /></article></section>
        <section class="panel intro"><div><span class="eyebrow">统一检测链路</span><h2>输入一个微博账号，获得两类风险结果</h2><p>系统抓取公开账号资料与近期微博，同时运行行为、文本、关系和传播智能体，输出账号风险与最近微博的信息风险。</p></div><button @click="view = 'weibo'">开始协同检测</button></section>
      </template>

      <template v-else>
        <section class="grid form-grid">
          <form class="panel" @submit.prevent="detectWeibo">
            <h2>微博公开数据抓取</h2>
            <p class="form-note">输入数字UID或包含数字UID的公开主页链接。系统默认匿名读取；若微博限制匿名访问，可由后端使用你本人登录会话的Cookie。</p>
            <label>微博UID或主页链接<input v-model.trim="weiboForm.target" required placeholder="例如：1642904381" /></label>
            <label>读取最近微博数<input v-model.number="weiboForm.recent_posts" type="number" min="1" max="100" /></label>
            <button :disabled="loading">{{ loading ? '抓取并分析中…' : '启动账号与信息协同检测' }}</button>
          </form>

          <article class="panel source-panel">
            <template v-if="weiboResult?.profile">
              <span class="eyebrow">实时数据摘要</span>
              <h2>{{ weiboResult.profile.screen_name || weiboResult.profile.uid }}</h2>
              <dl><dt>UID</dt><dd>{{ weiboResult.profile.uid }}</dd><dt>粉丝</dt><dd>{{ weiboResult.profile.followers_count.toLocaleString() }}</dd><dt>关注</dt><dd>{{ weiboResult.profile.follow_count.toLocaleString() }}</dd><dt>微博</dt><dd>{{ weiboResult.profile.statuses_count.toLocaleString() }}</dd><dt>认证</dt><dd>{{ weiboResult.profile.verified ? '是' : '否' }}</dd><dt>SVIP</dt><dd>{{ weiboResult.profile.svip ? '是' : '否' }}</dd><dt>会员等级</dt><dd>{{ weiboResult.profile.mbrank }}</dd><dt>会员类型</dt><dd>{{ weiboResult.profile.mbtype }}</dd><dt>本次读取</dt><dd>{{ weiboResult.profile.posts_collected }} 条</dd></dl>
              <div v-if="weiboResult.profile.latest_post" class="latest-post"><strong>最近微博</strong><p>{{ weiboResult.profile.latest_post.content }}</p><span>转发 {{ weiboResult.profile.latest_post.repost_count }} · 评论 {{ weiboResult.profile.latest_post.comment_count }} · 点赞 {{ weiboResult.profile.latest_post.attitude_count }}</span></div>
            </template>
            <p v-else>完成抓取后，这里将展示账号资料和用于信息检测的最近一条微博。</p>
          </article>
        </section>

        <InteractionGraph v-if="weiboResult?.interaction_graph" :graph="weiboResult.interaction_graph" />

        <section v-if="weiboResult" class="grid two detection-results">
          <ResultPanel title="账号综合风险" :result="weiboResult.account" />
          <ResultPanel v-if="weiboResult.information" title="最近微博信息风险" :result="weiboResult.information" />
          <article v-else class="panel"><h2>最近微博信息风险</h2><p class="form-note">未读取到可用于信息检测的公开微博。</p></article>
        </section>
        <p v-if="weiboResult" class="result-notice">{{ weiboResult.notice }}</p>
      </template>
    </main>
  </div>
</template>
