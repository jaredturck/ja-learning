import { useEffect, useRef, useState } from 'react'
import { ArrowLeft, Check, ChevronDown, ChevronRight, Lock, RotateCcw, Volume2 } from 'lucide-react'
import { levels } from './levels'
import type { QuizLevel, QuizSentence, SentenceChunk } from './levels'

interface ProgressState {
    completed_sentence_ids: string[]
}

interface QuizOption {
    japanese: string
    explanation: string
    is_correct: boolean
}

interface AudioLevel {
    file: string
    clips: Record<string, [number, number]>
}

interface AudioIndex {
    levels: Record<string, AudioLevel>
}

const progress_storage_key = 'nihongo-loop-progress'

const chunk_explanation_overrides: Record<string, string> = {
    '「おはよう」': '朝のあいさつ',
    '「お願いします」': '丁寧に依頼する定型表現',
    '「ここに来てください」': 'ここへ来るよう丁寧に頼む表現',
    '「さようなら」': '別れのあいさつ',
    '「すみません」': '謝罪や呼びかけに使う定型表現',
    '「コーヒーが好きです」': 'コーヒーが好きだと伝える表現',
    '「今日は休みです」': '今日は休みだと伝える表現',
    '「今日は寒いです」': '今日の寒さを伝える表現',
    '「名前を書いてください」': '名前を書くよう丁寧に頼む表現',
    '「待ってください」': '待つよう丁寧に頼む表現',
    '「日本に行きます」': '日本へ行くことを伝える表現',
    '「日本語を勉強します」': '日本語を勉強することを伝える表現',
    '「映画が好きです」': '映画が好きだと伝える表現',
    '「本を読みました」': '本を読んだことを伝える表現',
    '「本を読みます」': '本を読むことを伝える表現',
    '「東京に住んでいます」': '東京に住んでいることを伝える表現',
    '「水を飲みます」': '水を飲むことを伝える表現',
    '「水を飲んでください」': '水を飲むよう丁寧に頼む表現',
    '「猫がいます」': '猫がいることを伝える表現',
    '「駅に行きます」': '駅へ行くことを伝える表現',
    '「魚を食べません」': '魚を食べないことを伝える表現',
    あそこ: '話し手と聞き手から離れた場所を指す語',
    ありませんでした: '物が存在しなかったことを表す丁寧な形',
    い: 'い形容詞の語尾',
    いつも: '頻度が常にそうであることを表す語',
    いません: '人や動物が存在しないことを表す丁寧な形',
    いませんでした: '人や動物が存在しなかったことを表す丁寧な形',
    うどん: '麺料理を表す語',
    ことができます: '可能であることを表す丁寧な形',
    ご飯: '米飯や食事を表す語',
    しましょう: '一緒に行う提案を表す丁寧な形',
    そこ: '聞き手の近くの場所を指す語',
    たら: '条件や時を表す形',
    てはいけません: '禁止を表す丁寧な形',
    ても: '逆接の条件を表す形',
    てもいいです: '許可を表す丁寧な形',
    と言いました: '引用した内容を伝える表現',
    どの: 'どの名詞かを尋ねる語',
    どんな: '種類や性質を尋ねる語',
    ね: '同意や確認を求める終助詞',
    はずです: '当然の予想を表す丁寧な形',
    ほしいです: '物を欲しい気持ちを表す丁寧な形',
    よ: '強調や新しい情報を伝える終助詞',
    ジュース: '果汁飲料などを表す語',
    '一冊、': '本などを一冊数える表現',
    '三冊、': '本などを三冊数える表現',
    '二本、': '細長い物を二本数える表現',
    '二枚、': '薄く平たい物を二枚数える表現',
    作らない: '作るの普通体否定',
    勉強しない: '勉強するの普通体否定',
    午後三時: '午後の三時を表す時刻',
    古い: '物の古さを表すい形容詞',
    暑い: '気温の高さを表すい形容詞',
    書いた: '書くの普通体過去',
    書かない: '書くの普通体否定',
    書く: '文字や文章を記す動作',
    来ます: '来るの丁寧な現在・未来形',
    火曜日: '週の曜日の一つ',
    聞いた: '聞くの普通体過去',
    聞いて: '聞くのて形',
    聞かない: '聞くの普通体否定',
    聞く: '音を聞く、または質問する動作',
    行きました: '行くの丁寧な過去形',
    行きます: '行くの丁寧な現在・未来形',
    買わない: '買うの普通体否定',
    は: '主題を示す助詞',
    が: '主語を示す助詞',
    を: '目的語を示す助詞',
    に: '時点・行き先・存在場所などを示す助詞',
    で: '動作の場所や手段を示す助詞',
    と: '並列・相手・引用などを示す助詞',
    の: '所有や名詞同士のつながりを示す助詞',
    も: '追加や同類を示す助詞',
    へ: '方向を示す助詞',
    から: '起点・理由・順序などを示す助詞',
    まで: '終点や期限を示す助詞',
    か: '疑問を表す終助詞',
}

function get_chunk_explanation_lookup() {
    const explanation_counts = new Map<string, Map<string, number>>()

    levels.forEach((level) => {
        level.sentences.forEach((sentence) => {
            sentence.chunks.forEach((chunk) => {
                const chunk_counts = explanation_counts.get(chunk.japanese) ?? new Map<string, number>()
                chunk_counts.set(chunk.explanation, (chunk_counts.get(chunk.explanation) ?? 0) + 1)
                explanation_counts.set(chunk.japanese, chunk_counts)
            })
        })
    })

    return new Map(
        [...explanation_counts.entries()].map(([japanese, chunk_counts]) => {
            const explanation = chunk_explanation_overrides[japanese]
                ?? [...chunk_counts.entries()].sort((first, second) => second[1] - first[1])[0]?.[0]
                ?? ''
            return [japanese, explanation]
        }),
    )
}

const chunk_explanation_lookup = get_chunk_explanation_lookup()

function shuffle_items<T>(items: T[]) {
    const shuffled_items = [...items]

    for (let index = shuffled_items.length - 1; index > 0; index -= 1) {
        const random_index = Math.floor(Math.random() * (index + 1))
        const current_item = shuffled_items[index]
        shuffled_items[index] = shuffled_items[random_index]
        shuffled_items[random_index] = current_item
    }

    return shuffled_items
}

function get_saved_progress() {
    const saved_progress = localStorage.getItem(progress_storage_key)

    if (!saved_progress) {
        return { completed_sentence_ids: [] }
    }

    return JSON.parse(saved_progress) as ProgressState
}

function get_sentence_options(chunk: SentenceChunk) {
    const unique_distractors = [...new Set(chunk.distractors)].filter(
        (distractor) => distractor !== chunk.japanese,
    )
    const selected_distractors = shuffle_items(unique_distractors).slice(0, 3)
    const options: QuizOption[] = [
        { japanese: chunk.japanese, explanation: chunk.explanation, is_correct: true },
        ...selected_distractors.map((japanese) => ({
            japanese,
            explanation: chunk_explanation_lookup.get(japanese) ?? '',
            is_correct: false,
        })),
    ]

    return shuffle_items(options)
}

function get_completed_count(level: QuizLevel, completed_sentence_ids: string[]) {
    return level.sentences.filter((sentence) => completed_sentence_ids.includes(sentence.id)).length
}

function get_level_completed_sentence_ids(level: QuizLevel | undefined, completed_sentence_ids: string[]) {
    if (!level) {
        return []
    }

    return level.sentences
        .filter((sentence) => completed_sentence_ids.includes(sentence.id))
        .map((sentence) => sentence.id)
}

function get_sequential_sentence_order(level: QuizLevel) {
    return level.sentences.map((_, index) => index)
}

function get_sentence_order(level: QuizLevel | undefined, completed_sentence_ids: string[]) {
    if (!level) {
        return []
    }

    const incomplete_sentence_indexes = level.sentences
        .map((sentence, index) => ({ sentence, index }))
        .filter(({ sentence }) => !completed_sentence_ids.includes(sentence.id))
        .map(({ index }) => index)

    if (incomplete_sentence_indexes.length > 0) {
        return shuffle_items(incomplete_sentence_indexes)
    }

    return level.sentences.map((_, index) => index)
}

function get_initial_level_index(completed_sentence_ids: string[]) {
    const incomplete_level_index = levels.findIndex(
        (level) => get_completed_count(level, completed_sentence_ids) < level.sentences.length,
    )

    if (incomplete_level_index === -1) {
        return Math.max(levels.length - 1, 0)
    }

    return incomplete_level_index
}

function get_level_number(level_index: number) {
    return String(level_index + 1).padStart(2, '0')
}

function get_japanese_sentence(sentence: QuizSentence) {
    return sentence.chunks.map((chunk) => chunk.japanese).join('')
}

function get_sentence_text_size(sentence: QuizSentence) {
    if (sentence.chunks.length >= 11) {
        return 'clamp(1.45rem, 3.2vw, 2.7rem)'
    }

    if (sentence.chunks.length >= 8) {
        return 'clamp(1.7rem, 4vw, 3.25rem)'
    }

    return 'clamp(2.2rem, 5vw, 4.25rem)'
}

function get_option_text_size(japanese: string) {
    if (japanese.length >= 9) {
        return 'clamp(1.35rem, 3vw, 2.4rem)'
    }

    if (japanese.length >= 6) {
        return 'clamp(1.6rem, 3.8vw, 3rem)'
    }

    return 'clamp(2rem, 4.8vw, 3.8rem)'
}

function EmptyCurriculum() {
    return (
        <main className="flex h-dvh items-center justify-center bg-app px-6 text-center text-slate-100">
            <div>
                <p className="text-2xl font-semibold">レベルがありません</p>
                <p className="mt-2 text-slate-500">src/levels.ts に教材を追加してください。</p>
            </div>
        </main>
    )
}

function LevelStatus({ level, level_index, completed_count, is_sidebar_open, on_toggle_sidebar }: {
    level: QuizLevel
    level_index: number
    completed_count: number
    is_sidebar_open: boolean
    on_toggle_sidebar: () => void
}) {
    const level_progress = (completed_count / level.sentences.length) * 100

    return (
        <div className="relative z-30 shrink-0">
            <div className="flex items-center justify-between gap-4 text-sm">
                <button
                    aria-controls="course-sidebar"
                    aria-expanded={is_sidebar_open}
                    aria-label={`コース一覧を${is_sidebar_open ? '閉じる' : '開く'}`}
                    className="flex min-w-0 items-center gap-1.5 rounded-lg font-semibold text-violet-400 transition hover:text-violet-300 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-violet-400"
                    onClick={on_toggle_sidebar}
                    type="button"
                >
                    <span>レベル {get_level_number(level_index)}</span>
                    <ChevronDown
                        className={`transition-transform duration-200 ${is_sidebar_open ? 'rotate-180' : ''}`}
                        size={15}
                    />
                </button>
                <span className="shrink-0 tabular-nums text-slate-500">
                    {Math.min(completed_count + 1, level.sentences.length)} / {level.sentences.length}
                </span>
            </div>
            <div className="mt-3 h-1 overflow-hidden rounded-full bg-slate-800">
                <div
                    className="h-full rounded-full bg-violet-500 transition-[width] duration-300"
                    style={{ width: `${level_progress}%` }}
                />
            </div>
        </div>
    )
}

function CourseSidebar({ active_level_index, completed_sentence_ids, is_open, on_close, on_reset_progress, on_select_level }: {
    active_level_index: number
    completed_sentence_ids: string[]
    is_open: boolean
    on_close: () => void
    on_reset_progress: () => void
    on_select_level: (level_index: number) => void
}) {
    const active_level_ref = useRef<HTMLButtonElement | null>(null)
    const [is_reset_confirmation_open, set_is_reset_confirmation_open] = useState(false)
    const total_sentence_count = levels.reduce((total, level) => total + level.sentences.length, 0)
    const completed_sentence_count = levels.reduce(
        (total, level) => total + get_completed_count(level, completed_sentence_ids),
        0,
    )
    const completed_level_count = levels.filter(
        (level) => get_completed_count(level, completed_sentence_ids) === level.sentences.length,
    ).length
    const current_course_level_index = get_initial_level_index(completed_sentence_ids)
    const course_progress = total_sentence_count === 0
        ? 0
        : (completed_sentence_count / total_sentence_count) * 100

    useEffect(() => {
        if (is_open) {
            active_level_ref.current?.scrollIntoView({ block: 'center' })
        }
    }, [active_level_index, is_open])

    useEffect(() => {
        if (!is_open) {
            set_is_reset_confirmation_open(false)
        }
    }, [is_open])

    return (
        <>
            <button
                aria-label="コース一覧を閉じる"
                className={`fixed inset-x-0 bottom-0 top-16 z-10 bg-black/55 backdrop-blur-[2px] transition-opacity duration-200 ${
                    is_open ? 'opacity-100' : 'pointer-events-none opacity-0'
                }`}
                onClick={on_close}
                type="button"
            />
            <aside
                aria-hidden={!is_open}
                className={`fixed bottom-0 left-0 top-16 z-20 flex w-[min(24rem,calc(100vw-1rem))] flex-col border-r border-t border-slate-800 bg-slate-950/98 shadow-2xl shadow-black/40 transition-transform duration-200 ${
                    is_open ? 'pointer-events-auto translate-x-0' : 'pointer-events-none -translate-x-full'
                }`}
                id="course-sidebar"
                inert={!is_open}
            >
                <div className="shrink-0 border-b border-slate-800 p-5">
                    <div className="flex items-end justify-between gap-4">
                        <div>
                            <p className="text-xs font-semibold tracking-[0.16em] text-violet-400">コース進捗</p>
                            <p className="mt-1 text-3xl font-semibold tabular-nums text-white">
                                {Math.round(course_progress)}%
                            </p>
                        </div>
                        <p className="pb-1 text-sm tabular-nums text-slate-400">
                            {completed_level_count} / {levels.length} レベル
                        </p>
                    </div>
                    <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-800">
                        <div
                            className="h-full rounded-full bg-violet-500 transition-[width] duration-300"
                            style={{ width: `${course_progress}%` }}
                        />
                    </div>
                    <p className="mt-2 text-xs tabular-nums text-slate-500">
                        {completed_sentence_count} / {total_sentence_count} 文を完了
                    </p>
                </div>

                <div className="min-h-0 flex-1 overflow-y-scroll p-3">
                    <div className="space-y-2">
                        {levels.map((level, level_index) => {
                            const level_completed_count = get_completed_count(level, completed_sentence_ids)
                            const is_complete = level_completed_count === level.sentences.length
                            const is_current_course_level = level_index === current_course_level_index
                            const is_unlocked = is_complete || is_current_course_level
                            const is_active = level_index === active_level_index

                            return (
                                <button
                                    aria-current={is_active ? 'step' : undefined}
                                    className={`flex w-full items-start gap-3 rounded-2xl border p-3 text-left transition ${
                                        is_active
                                            ? 'border-violet-500 bg-violet-500/10'
                                            : is_unlocked
                                                ? 'border-slate-800 bg-slate-900/70 hover:border-slate-700 hover:bg-slate-900'
                                                : 'cursor-not-allowed border-transparent bg-slate-900/25 opacity-40'
                                    }`}
                                    disabled={!is_unlocked}
                                    key={level.id}
                                    onClick={() => on_select_level(level_index)}
                                    ref={is_active ? active_level_ref : null}
                                    type="button"
                                >
                                    <span className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl ${
                                        is_complete
                                            ? 'bg-emerald-400/10 text-emerald-300'
                                            : is_unlocked
                                                ? 'bg-violet-400/10 text-violet-300'
                                                : 'bg-slate-800 text-slate-500'
                                    }`}>
                                        {is_complete ? (
                                            <Check size={16} strokeWidth={2.5} />
                                        ) : is_unlocked ? (
                                            <ChevronRight size={17} />
                                        ) : (
                                            <Lock size={15} />
                                        )}
                                    </span>
                                    <span className="min-w-0 flex-1">
                                        <span className="flex items-center justify-between gap-3">
                                            <span className={`text-xs font-semibold tracking-[0.12em] ${
                                                is_active ? 'text-violet-300' : 'text-slate-500'
                                            }`}>
                                                レベル {get_level_number(level_index)}
                                            </span>
                                            <span className="shrink-0 text-xs tabular-nums text-slate-600">
                                                {level_completed_count} / {level.sentences.length}
                                            </span>
                                        </span>
                                        <span className={`mt-1 block font-semibold ${
                                            is_unlocked ? 'text-slate-100' : 'text-slate-500'
                                        }`}>
                                            {level.title}
                                        </span>
                                        <span className="mt-1 block text-xs leading-relaxed text-slate-500">
                                            {level.description}
                                        </span>
                                    </span>
                                </button>
                            )
                        })}
                    </div>
                </div>

                <div className="shrink-0 border-t border-slate-800 p-3">
                    <button
                        className="w-full rounded-xl border border-red-400/30 px-4 py-3 text-sm font-semibold text-red-300 transition hover:border-red-400/60 hover:bg-red-400/10 hover:text-red-200"
                        onClick={() => set_is_reset_confirmation_open(true)}
                        type="button"
                    >
                        進捗をリセット
                    </button>
                </div>
            </aside>

            {is_reset_confirmation_open ? (
                <div
                    className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 px-4 backdrop-blur-sm"
                    onClick={() => set_is_reset_confirmation_open(false)}
                    role="presentation"
                >
                    <div
                        aria-describedby="reset-progress-description"
                        aria-labelledby="reset-progress-title"
                        aria-modal="true"
                        className="w-full max-w-sm rounded-3xl border border-slate-700 bg-slate-950 p-6 shadow-2xl shadow-black/50"
                        onClick={(event) => event.stopPropagation()}
                        role="dialog"
                    >
                        <p className="text-xl font-semibold text-white" id="reset-progress-title">
                            進捗をリセットしますか？
                        </p>
                        <p className="mt-3 text-sm leading-relaxed text-slate-400" id="reset-progress-description">
                            すべてのレベルの進捗が削除され、レベル01から再開します。
                        </p>
                        <div className="mt-6 flex justify-end gap-3">
                            <button
                                className="rounded-xl border border-slate-700 px-4 py-2.5 text-sm font-semibold text-slate-300 transition hover:border-slate-600 hover:text-white"
                                onClick={() => set_is_reset_confirmation_open(false)}
                                type="button"
                            >
                                キャンセル
                            </button>
                            <button
                                className="rounded-xl bg-red-500 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-red-400"
                                onClick={on_reset_progress}
                                type="button"
                            >
                                リセット
                            </button>
                        </div>
                    </div>
                </div>
            ) : null}
        </>
    )
}

function SentenceBuilder({ sentence, current_chunk_index, is_audio_available, on_play_audio }: {
    sentence: QuizSentence
    current_chunk_index: number
    is_audio_available: boolean
    on_play_audio: () => void
}) {
    const text_size = get_sentence_text_size(sentence)

    return (
        <div
            aria-label="作成中の日本語文"
            className="flex min-h-0 w-full flex-1 items-center justify-center text-slate-100"
        >
            <div className="flex max-w-full items-center justify-center gap-3 sm:gap-4">
                <div className="flex min-w-0 max-w-full flex-wrap items-end justify-center gap-x-4 gap-y-3 sm:gap-x-6">
                    {sentence.chunks.map((chunk, chunk_index) => {
                        const is_complete = chunk_index < current_chunk_index
                        const is_current = chunk_index === current_chunk_index

                        return (
                            <div
                                className={`flex min-w-[2.25rem] flex-col items-center transition-opacity duration-150 ${
                                    is_complete ? 'animate-chunk-in opacity-100' : is_current ? 'opacity-100' : 'opacity-25'
                                }`}
                                key={`${chunk.japanese}-${chunk_index}`}
                            >
                                <span
                                    className={`font-japanese font-medium leading-none tracking-tight ${
                                        is_current ? 'text-violet-400' : ''
                                    }`}
                                    style={{ fontSize: text_size }}
                                >
                                    {is_complete ? chunk.japanese : '＿'}
                                </span>
                                <span className="mt-2 min-h-4 text-center text-xs text-slate-500 sm:text-sm">
                                    {is_complete ? chunk.explanation : ''}
                                </span>
                            </div>
                        )
                    })}
                </div>
                <button
                    aria-label="日本語の文を聞く"
                    className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-slate-700 text-slate-400 transition hover:border-violet-500 hover:text-violet-300 disabled:cursor-not-allowed disabled:opacity-25 disabled:hover:border-slate-700 disabled:hover:text-slate-400"
                    disabled={!is_audio_available}
                    onClick={on_play_audio}
                    type="button"
                >
                    <Volume2 size={18} />
                </button>
            </div>
        </div>
    )
}

function ChunkProgress({ current_chunk_index, total_chunks }: { current_chunk_index: number; total_chunks: number }) {
    return (
        <div className="flex h-1.5 w-full max-w-xl gap-1.5">
            {Array.from({ length: total_chunks }, (_, chunk_index) => (
                <span
                    className={`h-full flex-1 rounded-full transition-colors duration-150 ${
                        chunk_index < current_chunk_index
                            ? 'bg-violet-500'
                            : chunk_index === current_chunk_index
                                ? 'bg-violet-400/50'
                                : 'bg-slate-800'
                    }`}
                    key={chunk_index}
                />
            ))}
        </div>
    )
}

function OptionGrid({ options, feedback_japanese, feedback_state, is_locked, on_select_option }: {
    options: QuizOption[]
    feedback_japanese: string
    feedback_state: 'correct' | 'wrong' | 'complete' | null
    is_locked: boolean
    on_select_option: (option: QuizOption) => void
}) {
    return (
        <div className="grid h-[clamp(13rem,32vh,18rem)] shrink-0 grid-cols-2 gap-3 sm:gap-4">
            {options.map((option) => {
                const is_selected = option.japanese === feedback_japanese
                const is_revealed_correct = option.is_correct && feedback_state === 'wrong'
                const is_correct_feedback = (is_selected && feedback_state !== 'wrong') || is_revealed_correct
                const is_wrong_feedback = is_selected && feedback_state === 'wrong'
                const show_explanation = feedback_state === 'wrong' && (is_wrong_feedback || is_revealed_correct)

                return (
                    <button
                        aria-label={`選択肢：${option.japanese}`}
                        className={`group relative min-h-0 min-w-0 overflow-hidden rounded-2xl border px-3 transition duration-150 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-violet-400 sm:rounded-3xl sm:px-6 ${
                            is_correct_feedback
                                ? 'border-emerald-400 bg-emerald-400/12 text-emerald-200'
                                : is_wrong_feedback
                                    ? 'border-red-400 bg-red-400/12 text-red-200'
                                    : 'border-slate-700 bg-slate-900/85 text-slate-100 hover:border-violet-500 hover:bg-slate-900 active:scale-[0.985]'
                        } ${is_locked && !is_selected && !is_revealed_correct ? 'opacity-45' : ''}`}
                        disabled={is_locked}
                        key={option.japanese}
                        onClick={() => on_select_option(option)}
                        type="button"
                    >
                        <span className="flex h-full flex-col items-center justify-center">
                            <span
                                className="font-japanese block break-all font-medium leading-tight tracking-tight"
                                style={{ fontSize: get_option_text_size(option.japanese) }}
                            >
                                {option.japanese}
                            </span>
                            <span
                                className={`mt-2 min-h-5 text-center text-sm font-medium transition-opacity duration-150 sm:text-base ${
                                    show_explanation ? 'opacity-100' : 'opacity-0'
                                }`}
                            >
                                {show_explanation ? option.explanation : ''}
                            </span>
                        </span>
                    </button>
                )
            })}
        </div>
    )
}

function LevelComplete({ level_index, has_next_level, on_continue, on_replay }: {
    level_index: number
    has_next_level: boolean
    on_continue: () => void
    on_replay: () => void
}) {
    return (
        <div className="flex min-h-0 flex-1 flex-col items-center justify-center text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-emerald-400/10 text-emerald-300">
                <Check size={32} strokeWidth={2.5} />
            </div>
            <p className="mt-6 text-sm font-semibold tracking-[0.18em] text-violet-400">
                レベル {get_level_number(level_index)} 完了
            </p>
            <p className="mt-3 text-3xl font-semibold text-slate-100 sm:text-5xl">
                {has_next_level ? '次のレベルへ進めます。' : 'すべてのレベルを完了しました。'}
            </p>
            <div className="mt-8 flex flex-col-reverse gap-3 sm:flex-row">
                <button
                    className="rounded-2xl border border-slate-700 px-8 py-4 text-base font-semibold text-slate-200 transition hover:border-violet-500 hover:text-white active:scale-[0.98]"
                    onClick={on_replay}
                    type="button"
                >
                    もう一度
                </button>
                {has_next_level ? (
                    <button
                        className="rounded-2xl bg-violet-500 px-8 py-4 text-base font-semibold text-white transition hover:bg-violet-400 active:scale-[0.98]"
                        onClick={on_continue}
                        type="button"
                    >
                        次へ
                    </button>
                ) : null}
            </div>
        </div>
    )
}

function App() {
    const [progress, set_progress] = useState<ProgressState>(get_saved_progress)
    const [active_level_index, set_active_level_index] = useState(() => get_initial_level_index(progress.completed_sentence_ids))
    const [level_completed_sentence_ids, set_level_completed_sentence_ids] = useState(() =>
        get_level_completed_sentence_ids(levels[active_level_index], progress.completed_sentence_ids),
    )
    const [sentence_order, set_sentence_order] = useState<number[]>(() =>
        get_sentence_order(levels[active_level_index], level_completed_sentence_ids),
    )
    const [sentence_order_index, set_sentence_order_index] = useState(0)
    const [current_chunk_index, set_current_chunk_index] = useState(0)
    const [options, set_options] = useState<QuizOption[]>([])
    const [feedback_japanese, set_feedback_japanese] = useState('')
    const [feedback_state, set_feedback_state] = useState<'correct' | 'wrong' | 'complete' | null>(null)
    const [is_locked, set_is_locked] = useState(false)
    const [option_seed, set_option_seed] = useState(0)
    const [is_sidebar_open, set_is_sidebar_open] = useState(false)
    const [audio_index, set_audio_index] = useState<AudioIndex | null>(null)
    const [is_level_audio_ready, set_is_level_audio_ready] = useState(false)
    const feedback_timeout = useRef<number | null>(null)
    const audio_context = useRef<AudioContext | null>(null)
    const audio_buffer = useRef<AudioBuffer | null>(null)
    const audio_source = useRef<AudioBufferSourceNode | null>(null)
    const loaded_audio_level_id = useRef('')

    const active_level = levels[active_level_index]
    const sentence_index = sentence_order[sentence_order_index] ?? 0
    const active_sentence = active_level?.sentences[sentence_index]
    const active_sentence_text = active_sentence ? get_japanese_sentence(active_sentence) : ''
    const completed_count = active_level ? get_completed_count(active_level, level_completed_sentence_ids) : 0
    const is_level_complete = Boolean(active_level && completed_count === active_level.sentences.length)
    const can_play_sentence_audio = Boolean(
        is_level_audio_ready
        && audio_index?.levels[active_level.id]?.clips[active_sentence_text],
    )

    useEffect(() => {
        localStorage.setItem(progress_storage_key, JSON.stringify(progress))
    }, [progress])

    useEffect(() => {
        let is_cancelled = false

        fetch(`${import.meta.env.BASE_URL}audio/index.json`)
            .then((response) => response.json() as Promise<AudioIndex>)
            .then((next_audio_index) => {
                if (!is_cancelled) {
                    set_audio_index(next_audio_index)
                }
            })
            .catch(() => undefined)

        return () => {
            is_cancelled = true
        }
    }, [])

    useEffect(() => {
        let is_cancelled = false
        const level_audio = audio_index?.levels[active_level.id]

        audio_source.current?.stop()
        audio_source.current = null
        audio_buffer.current = null
        loaded_audio_level_id.current = ''
        set_is_level_audio_ready(false)

        if (!level_audio) {
            return () => {
                is_cancelled = true
            }
        }

        const context = audio_context.current ?? new AudioContext()
        audio_context.current = context

        fetch(`${import.meta.env.BASE_URL}audio/${level_audio.file}`)
            .then((response) => response.arrayBuffer())
            .then((audio_data) => context.decodeAudioData(audio_data))
            .then((next_audio_buffer) => {
                if (!is_cancelled) {
                    audio_buffer.current = next_audio_buffer
                    loaded_audio_level_id.current = active_level.id
                    set_is_level_audio_ready(true)
                }
            })
            .catch(() => undefined)

        return () => {
            is_cancelled = true
        }
    }, [active_level.id, audio_index])

    useEffect(() => {
        return () => {
            if (feedback_timeout.current !== null) {
                window.clearTimeout(feedback_timeout.current)
            }

            audio_source.current?.stop()
            audio_context.current?.close()
        }
    }, [])

    useEffect(() => {
        if (!active_sentence || !active_sentence.chunks[current_chunk_index]) {
            set_options([])
            return
        }

        set_options(get_sentence_options(active_sentence.chunks[current_chunk_index]))
    }, [active_sentence, current_chunk_index, option_seed])

    const play_audio = (text: string) => {
        const clip = audio_index?.levels[active_level.id]?.clips[text]
        const context = audio_context.current
        const buffer = audio_buffer.current

        if (!clip || !context || !buffer || loaded_audio_level_id.current !== active_level.id) {
            return
        }

        audio_source.current?.stop()

        const source = context.createBufferSource()
        source.buffer = buffer
        source.connect(context.destination)
        audio_source.current = source

        context.resume().then(() => {
            if (audio_source.current === source) {
                source.start(0, clip[0], clip[1])
            }
        })
    }

    const handle_select_option = (option: QuizOption) => {
        if (!active_sentence || is_locked) {
            return
        }

        play_audio(option.japanese)
        set_is_locked(true)
        set_feedback_japanese(option.japanese)

        if (!option.is_correct) {
            set_feedback_state('wrong')

            feedback_timeout.current = window.setTimeout(() => {
                set_current_chunk_index(0)
                set_option_seed((current_seed) => current_seed + 1)
                set_feedback_japanese('')
                set_feedback_state(null)
                set_is_locked(false)
            }, 3000)
            return
        }

        const next_chunk_index = current_chunk_index + 1
        const sentence_is_complete = next_chunk_index === active_sentence.chunks.length

        set_current_chunk_index(next_chunk_index)
        set_feedback_state(sentence_is_complete ? 'complete' : 'correct')

        if (sentence_is_complete) {
            set_level_completed_sentence_ids((current_sentence_ids) => [
                ...new Set([...current_sentence_ids, active_sentence.id]),
            ])
            set_progress((current_progress) => ({
                completed_sentence_ids: [...new Set([...current_progress.completed_sentence_ids, active_sentence.id])],
            }))

            feedback_timeout.current = window.setTimeout(() => {
                set_sentence_order_index((current_index) => Math.min(current_index + 1, sentence_order.length - 1))
                set_current_chunk_index(0)
                set_option_seed((current_seed) => current_seed + 1)
                set_feedback_japanese('')
                set_feedback_state(null)
                set_is_locked(false)
            }, 700)
            return
        }

        feedback_timeout.current = window.setTimeout(() => {
            set_feedback_japanese('')
            set_feedback_state(null)
            set_is_locked(false)
        }, 180)
    }

    useEffect(() => {
        const handle_key_down = (event: KeyboardEvent) => {
            if (event.key === 'Escape' && is_sidebar_open) {
                set_is_sidebar_open(false)
            }
        }

        window.addEventListener('keydown', handle_key_down)
        return () => window.removeEventListener('keydown', handle_key_down)
    }, [is_sidebar_open])

    const handle_restart_sentence = () => {
        if (feedback_timeout.current !== null) {
            window.clearTimeout(feedback_timeout.current)
            feedback_timeout.current = null
        }

        set_current_chunk_index(0)
        set_option_seed((current_seed) => current_seed + 1)
        set_feedback_japanese('')
        set_feedback_state(null)
        set_is_locked(false)
    }

    const handle_previous_level = () => {
        if (active_level_index === 0) {
            return
        }

        if (feedback_timeout.current !== null) {
            window.clearTimeout(feedback_timeout.current)
            feedback_timeout.current = null
        }

        const previous_level_index = active_level_index - 1
        const previous_level = levels[previous_level_index]

        set_active_level_index(previous_level_index)
        set_level_completed_sentence_ids([])
        set_sentence_order(get_sequential_sentence_order(previous_level))
        set_sentence_order_index(0)
        set_current_chunk_index(0)
        set_option_seed((current_seed) => current_seed + 1)
        set_feedback_japanese('')
        set_feedback_state(null)
        set_is_locked(false)
    }

    const handle_next_level = () => {
        const next_level_index = Math.min(active_level_index + 1, levels.length - 1)
        const next_level = levels[next_level_index]
        const saved_sentence_ids = get_level_completed_sentence_ids(next_level, progress.completed_sentence_ids)
        const should_replay_level = saved_sentence_ids.length === next_level.sentences.length
        const next_sentence_ids = should_replay_level ? [] : saved_sentence_ids

        set_active_level_index(next_level_index)
        set_level_completed_sentence_ids(next_sentence_ids)
        set_sentence_order(
            should_replay_level
                ? get_sequential_sentence_order(next_level)
                : get_sentence_order(next_level, next_sentence_ids),
        )
        set_sentence_order_index(0)
        set_current_chunk_index(0)
        set_option_seed((current_seed) => current_seed + 1)
        set_feedback_japanese('')
        set_feedback_state(null)
        set_is_locked(false)
    }

    const handle_select_level = (level_index: number) => {
        if (feedback_timeout.current !== null) {
            window.clearTimeout(feedback_timeout.current)
            feedback_timeout.current = null
        }

        const selected_level = levels[level_index]
        const saved_sentence_ids = get_level_completed_sentence_ids(selected_level, progress.completed_sentence_ids)
        const should_replay_level = saved_sentence_ids.length === selected_level.sentences.length
        const selected_sentence_ids = should_replay_level ? [] : saved_sentence_ids

        set_active_level_index(level_index)
        set_level_completed_sentence_ids(selected_sentence_ids)
        set_sentence_order(
            should_replay_level
                ? get_sequential_sentence_order(selected_level)
                : get_sentence_order(selected_level, selected_sentence_ids),
        )
        set_sentence_order_index(0)
        set_current_chunk_index(0)
        set_option_seed((current_seed) => current_seed + 1)
        set_feedback_japanese('')
        set_feedback_state(null)
        set_is_locked(false)
        set_is_sidebar_open(false)
    }

    const handle_replay_level = () => {
        if (feedback_timeout.current !== null) {
            window.clearTimeout(feedback_timeout.current)
            feedback_timeout.current = null
        }

        const active_sentence_ids = new Set(active_level.sentences.map((sentence) => sentence.id))
        const next_progress = {
            completed_sentence_ids: progress.completed_sentence_ids.filter(
                (sentence_id) => !active_sentence_ids.has(sentence_id),
            ),
        }

        localStorage.setItem(progress_storage_key, JSON.stringify(next_progress))
        set_progress(next_progress)
        set_level_completed_sentence_ids([])
        set_sentence_order(get_sentence_order(active_level, []))
        set_sentence_order_index(0)
        set_current_chunk_index(0)
        set_option_seed((current_seed) => current_seed + 1)
        set_feedback_japanese('')
        set_feedback_state(null)
        set_is_locked(false)
    }

    const handle_reset_progress = () => {
        if (feedback_timeout.current !== null) {
            window.clearTimeout(feedback_timeout.current)
            feedback_timeout.current = null
        }

        const next_progress = { completed_sentence_ids: [] }
        const first_level = levels[0]

        localStorage.removeItem(progress_storage_key)
        set_progress(next_progress)
        set_active_level_index(0)
        set_level_completed_sentence_ids([])
        set_sentence_order(get_sentence_order(first_level, []))
        set_sentence_order_index(0)
        set_current_chunk_index(0)
        set_option_seed((current_seed) => current_seed + 1)
        set_feedback_japanese('')
        set_feedback_state(null)
        set_is_locked(false)
        set_is_sidebar_open(false)
    }

    if (levels.length === 0) {
        return <EmptyCurriculum />
    }

    return (
        <div className="h-dvh overflow-hidden bg-app text-slate-100">
            <CourseSidebar
                active_level_index={active_level_index}
                completed_sentence_ids={progress.completed_sentence_ids}
                is_open={is_sidebar_open}
                on_close={() => set_is_sidebar_open(false)}
                on_reset_progress={handle_reset_progress}
                on_select_level={handle_select_level}
            />

            <main className="mx-auto flex h-full w-full max-w-6xl flex-col px-4 py-4 sm:px-8 sm:py-6">
                <LevelStatus
                    completed_count={completed_count}
                    is_sidebar_open={is_sidebar_open}
                    level={active_level}
                    level_index={active_level_index}
                    on_toggle_sidebar={() => set_is_sidebar_open((current_state) => !current_state)}
                />

                {is_level_complete ? (
                    <LevelComplete
                        has_next_level={active_level_index < levels.length - 1}
                        level_index={active_level_index}
                        on_continue={handle_next_level}
                        on_replay={handle_replay_level}
                    />
                ) : active_sentence ? (
                    <>
                        <div className="relative mt-4 shrink-0 border-b border-slate-800 pb-4 text-center sm:mt-5 sm:pb-5">
                            <button
                                aria-label="前のレベル"
                                className="absolute left-0 top-1/2 flex h-10 w-10 -translate-y-1/2 items-center justify-center rounded-xl border border-slate-700 text-slate-500 transition hover:border-slate-600 hover:text-slate-200 disabled:cursor-not-allowed disabled:opacity-20 disabled:hover:border-slate-700 disabled:hover:text-slate-500"
                                disabled={active_level_index === 0}
                                onClick={handle_previous_level}
                                type="button"
                            >
                                <ArrowLeft size={19} />
                            </button>
                            <h1 className="px-14 text-[clamp(1.65rem,3.5vw,3rem)] font-semibold leading-tight tracking-[-0.035em] text-white">
                                {active_sentence.english}
                            </h1>
                            <button
                                aria-label="文を最初からやり直す"
                                className="absolute right-0 top-1/2 flex h-10 w-10 -translate-y-1/2 items-center justify-center rounded-xl border border-slate-700 text-slate-500 transition hover:border-slate-600 hover:text-slate-200"
                                onClick={handle_restart_sentence}
                                type="button"
                            >
                                <RotateCcw size={18} />
                            </button>
                        </div>

                        <div className="flex min-h-0 flex-1 flex-col items-center justify-center py-3 sm:py-5">
                            <SentenceBuilder
                                current_chunk_index={current_chunk_index}
                                is_audio_available={can_play_sentence_audio}
                                on_play_audio={() => play_audio(active_sentence_text)}
                                sentence={active_sentence}
                            />
                            <ChunkProgress
                                current_chunk_index={current_chunk_index}
                                total_chunks={active_sentence.chunks.length}
                            />
                        </div>

                        <OptionGrid
                            feedback_japanese={feedback_japanese}
                            feedback_state={feedback_state}
                            is_locked={is_locked}
                            on_select_option={handle_select_option}
                            options={options}
                        />
                    </>
                ) : null}
            </main>
        </div>
    )
}

export default App
