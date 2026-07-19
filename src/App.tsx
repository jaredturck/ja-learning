import { useEffect, useRef, useState } from 'react'
import { Check, RotateCcw } from 'lucide-react'
import { levels } from './levels'
import type { QuizLevel, QuizSentence, SentenceChunk } from './levels'

interface ProgressState {
    completed_sentence_ids: string[]
}

interface QuizOption {
    japanese: string
    is_correct: boolean
}

const progress_storage_key = 'nihongo-loop-progress'

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
        { japanese: chunk.japanese, is_correct: true },
        ...selected_distractors.map((japanese) => ({ japanese, is_correct: false })),
    ]

    return shuffle_items(options)
}

function get_completed_count(level: QuizLevel, completed_sentence_ids: string[]) {
    return level.sentences.filter((sentence) => completed_sentence_ids.includes(sentence.id)).length
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
                <p className="text-2xl font-semibold">No levels found</p>
                <p className="mt-2 text-slate-500">Add curriculum data to src/levels.ts.</p>
            </div>
        </main>
    )
}

function LevelStatus({ level, level_index, completed_count }: {
    level: QuizLevel
    level_index: number
    completed_count: number
}) {
    const level_progress = (completed_count / level.sentences.length) * 100

    return (
        <div className="shrink-0">
            <div className="flex items-center justify-between gap-4 text-sm">
                <div className="min-w-0">
                    <span className="font-semibold text-violet-400">LEVEL {get_level_number(level_index)}</span>
                    <span className="mx-2 text-slate-700">/</span>
                    <span className="truncate text-slate-400">{level.title}</span>
                </div>
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

function SentenceBuilder({ sentence, current_chunk_index }: {
    sentence: QuizSentence
    current_chunk_index: number
}) {
    const text_size = get_sentence_text_size(sentence)

    return (
        <div
            aria-label="Japanese sentence under construction"
            className="flex min-h-0 w-full flex-1 items-center justify-center text-slate-100"
        >
            <div className="flex max-w-full flex-wrap items-end justify-center gap-x-4 gap-y-3 sm:gap-x-6">
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
                                {is_complete ? chunk.english : ''}
                            </span>
                        </div>
                    )
                })}
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
            {options.map((option, option_index) => {
                const is_selected = option.japanese === feedback_japanese
                const is_revealed_correct = option.is_correct && feedback_state === 'wrong'
                const is_correct_feedback = (is_selected && feedback_state !== 'wrong') || is_revealed_correct
                const is_wrong_feedback = is_selected && feedback_state === 'wrong'

                return (
                    <button
                        aria-label={`Option ${option_index + 1}: ${option.japanese}`}
                        className={`group relative min-h-0 min-w-0 overflow-hidden rounded-2xl border px-3 transition duration-150 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-violet-400 sm:rounded-3xl sm:px-6 ${
                            is_correct_feedback
                                ? 'border-emerald-400 bg-emerald-400/12 text-emerald-200'
                                : is_wrong_feedback
                                    ? 'border-red-400 bg-red-400/12 text-red-200'
                                    : 'border-slate-700 bg-slate-900/85 text-slate-100 hover:border-violet-500 hover:bg-slate-900 active:scale-[0.985]'
                        } ${is_locked && !is_selected && !is_revealed_correct ? 'opacity-45' : ''}`}
                        disabled={is_locked}
                        key={`${option.japanese}-${option_index}`}
                        onClick={() => on_select_option(option)}
                        type="button"
                    >
                        <span className="absolute left-3 top-3 flex h-6 w-6 items-center justify-center rounded-md border border-slate-700 text-xs font-medium text-slate-500 sm:left-4 sm:top-4">
                            {option_index + 1}
                        </span>
                        <span
                            className="font-japanese block break-all font-medium leading-tight tracking-tight"
                            style={{ fontSize: get_option_text_size(option.japanese) }}
                        >
                            {option.japanese}
                        </span>
                    </button>
                )
            })}
        </div>
    )
}

function LevelComplete({ level_index, has_next_level, on_continue }: {
    level_index: number
    has_next_level: boolean
    on_continue: () => void
}) {
    return (
        <div className="flex min-h-0 flex-1 flex-col items-center justify-center text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-emerald-400/10 text-emerald-300">
                <Check size={32} strokeWidth={2.5} />
            </div>
            <p className="mt-6 text-sm font-semibold tracking-[0.18em] text-violet-400">
                LEVEL {get_level_number(level_index)} COMPLETE
            </p>
            <p className="mt-3 text-3xl font-semibold text-slate-100 sm:text-5xl">
                {has_next_level ? 'Next level ready.' : 'Curriculum complete.'}
            </p>
            {has_next_level ? (
                <button
                    className="mt-8 rounded-2xl bg-violet-500 px-8 py-4 text-base font-semibold text-white transition hover:bg-violet-400 active:scale-[0.98]"
                    onClick={on_continue}
                    type="button"
                >
                    Continue
                </button>
            ) : null}
        </div>
    )
}

function App() {
    const [progress, set_progress] = useState<ProgressState>(get_saved_progress)
    const [active_level_index, set_active_level_index] = useState(() => get_initial_level_index(progress.completed_sentence_ids))
    const [sentence_order, set_sentence_order] = useState<number[]>(() => get_sentence_order(levels[active_level_index], progress.completed_sentence_ids))
    const [sentence_order_index, set_sentence_order_index] = useState(0)
    const [current_chunk_index, set_current_chunk_index] = useState(0)
    const [options, set_options] = useState<QuizOption[]>([])
    const [feedback_japanese, set_feedback_japanese] = useState('')
    const [feedback_state, set_feedback_state] = useState<'correct' | 'wrong' | 'complete' | null>(null)
    const [is_locked, set_is_locked] = useState(false)
    const [option_seed, set_option_seed] = useState(0)
    const feedback_timeout = useRef<number | null>(null)

    const active_level = levels[active_level_index]
    const sentence_index = sentence_order[sentence_order_index] ?? 0
    const active_sentence = active_level?.sentences[sentence_index]
    const completed_count = active_level ? get_completed_count(active_level, progress.completed_sentence_ids) : 0
    const is_level_complete = Boolean(active_level && completed_count === active_level.sentences.length)

    useEffect(() => {
        localStorage.setItem(progress_storage_key, JSON.stringify(progress))
    }, [progress])

    useEffect(() => {
        return () => {
            if (feedback_timeout.current !== null) {
                window.clearTimeout(feedback_timeout.current)
            }
        }
    }, [])

    useEffect(() => {
        if (!active_sentence || !active_sentence.chunks[current_chunk_index]) {
            set_options([])
            return
        }

        set_options(get_sentence_options(active_sentence.chunks[current_chunk_index]))
    }, [active_sentence, current_chunk_index, option_seed])

    const handle_select_option = (option: QuizOption) => {
        if (!active_sentence || is_locked) {
            return
        }

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
            }, 1000)
            return
        }

        const next_chunk_index = current_chunk_index + 1
        const sentence_is_complete = next_chunk_index === active_sentence.chunks.length

        set_current_chunk_index(next_chunk_index)
        set_feedback_state(sentence_is_complete ? 'complete' : 'correct')

        if (sentence_is_complete) {
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
            const option_index = Number(event.key) - 1

            if (option_index < 0 || option_index > 3 || !options[option_index] || is_locked) {
                return
            }

            handle_select_option(options[option_index])
        }

        window.addEventListener('keydown', handle_key_down)
        return () => window.removeEventListener('keydown', handle_key_down)
    })

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

    const handle_next_level = () => {
        const next_level_index = Math.min(active_level_index + 1, levels.length - 1)

        set_active_level_index(next_level_index)
        set_sentence_order(get_sentence_order(levels[next_level_index], progress.completed_sentence_ids))
        set_sentence_order_index(0)
        set_current_chunk_index(0)
        set_option_seed((current_seed) => current_seed + 1)
        set_feedback_japanese('')
        set_feedback_state(null)
        set_is_locked(false)
    }

    if (levels.length === 0) {
        return <EmptyCurriculum />
    }

    return (
        <div className="h-dvh overflow-hidden bg-app text-slate-100">
            <main className="mx-auto flex h-full w-full max-w-6xl flex-col px-4 py-4 sm:px-8 sm:py-6">
                <LevelStatus
                    completed_count={completed_count}
                    level={active_level}
                    level_index={active_level_index}
                />

                {is_level_complete ? (
                    <LevelComplete
                        has_next_level={active_level_index < levels.length - 1}
                        level_index={active_level_index}
                        on_continue={handle_next_level}
                    />
                ) : active_sentence ? (
                    <>
                        <div className="relative mt-4 shrink-0 border-b border-slate-800 pb-4 text-center sm:mt-5 sm:pb-5">
                            <h1 className="text-[clamp(1.65rem,3.5vw,3rem)] font-semibold leading-tight tracking-[-0.035em] text-white">
                                {active_sentence.english}
                            </h1>
                            <button
                                aria-label="Restart sentence"
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
