import { useEffect, useMemo, useRef, useState } from 'react'
import {
    ArrowRight,
    BookOpen,
    Check,
    ChevronRight,
    CircleHelp,
    Flame,
    Keyboard,
    Layers3,
    RotateCcw,
    Sparkles,
    Target,
    Trophy,
    X,
} from 'lucide-react'
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

function get_level_number(level_index: number) {
    return String(level_index + 1).padStart(2, '0')
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

function EmptyCurriculum() {
    return (
        <main className="mx-auto flex min-h-[calc(100vh-5rem)] w-full max-w-7xl items-center px-5 py-10 sm:px-8 lg:px-10">
            <section className="grid w-full overflow-hidden rounded-[2rem] border border-slate-200/80 bg-white shadow-[0_30px_90px_-45px_rgba(15,23,42,0.35)] lg:grid-cols-[0.9fr_1.1fr]">
                <div className="relative overflow-hidden bg-slate-950 px-7 py-10 text-white sm:px-10 sm:py-14">
                    <div className="absolute -left-24 -top-24 h-72 w-72 rounded-full bg-violet-500/20 blur-3xl" />
                    <div className="absolute -bottom-32 -right-20 h-80 w-80 rounded-full bg-cyan-400/15 blur-3xl" />

                    <div className="relative">
                        <div className="mb-10 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] text-slate-300">
                            <Sparkles size={14} />
                            Interface complete
                        </div>

                        <h1 className="max-w-lg text-4xl font-semibold tracking-[-0.04em] sm:text-5xl">
                            Your Japanese curriculum is ready to be plugged in.
                        </h1>
                        <p className="mt-5 max-w-md text-base leading-7 text-slate-300">
                            The quiz engine, reset loop, keyboard controls, progress tracking, and responsive interface are all wired up. The level data is intentionally empty.
                        </p>

                        <div className="mt-10 grid max-w-md grid-cols-3 gap-3">
                            <FeatureStat icon={<Target size={18} />} label="Perfect runs" />
                            <FeatureStat icon={<Keyboard size={18} />} label="Keys 1–4" />
                            <FeatureStat icon={<Layers3 size={18} />} label="Level ready" />
                        </div>
                    </div>
                </div>

                <div className="flex flex-col justify-center px-7 py-10 sm:px-10 sm:py-14">
                    <div className="mb-8 flex h-12 w-12 items-center justify-center rounded-2xl bg-violet-100 text-violet-700">
                        <BookOpen size={23} />
                    </div>
                    <p className="text-sm font-semibold uppercase tracking-[0.16em] text-violet-700">Next step</p>
                    <h2 className="mt-2 text-3xl font-semibold tracking-[-0.03em] text-slate-950">
                        Add levels to <code className="font-mono text-[0.88em] text-violet-700">src/levels.ts</code>
                    </h2>
                    <p className="mt-4 max-w-xl leading-7 text-slate-600">
                        Once the exported <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-sm">levels</code> array contains data, this screen automatically becomes the full quiz experience. No application logic needs changing.
                    </p>

                    <div className="mt-8 rounded-2xl border border-slate-200 bg-slate-50 p-5">
                        <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
                            <span>Expected structure</span>
                            <span>TypeScript</span>
                        </div>
                        <pre className="mt-4 overflow-x-auto text-sm leading-6 text-slate-700"><code>{`level → sentences → chunks\nchunk → japanese + english + distractors`}</code></pre>
                    </div>
                </div>
            </section>
        </main>
    )
}

function FeatureStat({ icon, label }: { icon: React.ReactNode; label: string }) {
    return (
        <div className="rounded-2xl border border-white/10 bg-white/[0.06] p-4">
            <div className="text-cyan-300">{icon}</div>
            <p className="mt-3 text-xs font-medium leading-5 text-slate-300">{label}</p>
        </div>
    )
}

function AppHeader({ streak }: { streak: number }) {
    return (
        <header className="border-b border-slate-200/70 bg-white/75 backdrop-blur-xl">
            <div className="mx-auto flex h-20 max-w-7xl items-center justify-between px-5 sm:px-8 lg:px-10">
                <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-950 text-white shadow-lg shadow-slate-950/15">
                        <span className="font-japanese text-lg font-semibold">文</span>
                    </div>
                    <div>
                        <p className="text-base font-semibold tracking-[-0.02em] text-slate-950">Nihongo Loop</p>
                        <p className="text-xs text-slate-500">Build intuition, one chunk at a time</p>
                    </div>
                </div>

                <div className="flex items-center gap-2 rounded-full border border-orange-200 bg-orange-50 px-3.5 py-2 text-sm font-semibold text-orange-700">
                    <Flame size={17} fill="currentColor" />
                    <span>{streak} streak</span>
                </div>
            </div>
        </header>
    )
}

function LevelRail({ active_level_index, completed_sentence_ids, on_select_level }: {
    active_level_index: number
    completed_sentence_ids: string[]
    on_select_level: (level_index: number) => void
}) {
    return (
        <aside className="hidden w-72 shrink-0 lg:block">
            <div className="sticky top-6 max-h-[calc(100vh-3rem)] overflow-y-auto rounded-[1.75rem] border border-slate-200/80 bg-white p-4 shadow-[0_24px_70px_-48px_rgba(15,23,42,0.35)]">
                <div className="px-3 pb-4 pt-2">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Curriculum</p>
                    <h2 className="mt-1 text-xl font-semibold tracking-[-0.02em] text-slate-950">Learning path</h2>
                </div>

                <div className="space-y-1.5">
                    {levels.map((level, level_index) => {
                        const completed_count = get_completed_count(level, completed_sentence_ids)
                        const is_complete = completed_count === level.sentences.length
                        const is_active = level_index === active_level_index

                        return (
                            <button
                                className={`group flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-left transition ${
                                    is_active
                                        ? 'bg-slate-950 text-white shadow-lg shadow-slate-950/15'
                                        : 'text-slate-600 hover:bg-slate-100 hover:text-slate-950'
                                }`}
                                key={level.id}
                                onClick={() => on_select_level(level_index)}
                                type="button"
                            >
                                <span className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-xs font-bold ${
                                    is_active ? 'bg-white/10 text-white' : 'bg-slate-100 text-slate-500 group-hover:bg-white'
                                }`}>
                                    {is_complete ? <Check size={17} /> : get_level_number(level_index)}
                                </span>
                                <span className="min-w-0 flex-1">
                                    <span className="block truncate text-sm font-semibold">{level.title}</span>
                                    <span className={`mt-0.5 block text-xs ${is_active ? 'text-slate-400' : 'text-slate-400'}`}>
                                        {completed_count} / {level.sentences.length} sentences
                                    </span>
                                </span>
                                <ChevronRight className={is_active ? 'text-slate-500' : 'text-slate-300'} size={17} />
                            </button>
                        )
                    })}
                </div>
            </div>
        </aside>
    )
}

function MobileLevelSelect({ active_level_index, on_select_level }: { active_level_index: number; on_select_level: (level_index: number) => void }) {
    return (
        <div className="mb-4 flex items-center gap-3 rounded-2xl border border-slate-200/80 bg-white px-4 py-3 shadow-[0_20px_60px_-48px_rgba(15,23,42,0.3)] lg:hidden">
            <span className="text-xs font-bold uppercase tracking-[0.14em] text-slate-400">Level</span>
            <select
                className="min-w-0 flex-1 bg-transparent text-sm font-semibold text-slate-900 outline-none"
                onChange={(event) => on_select_level(Number(event.target.value))}
                value={active_level_index}
            >
                {levels.map((level, level_index) => (
                    <option key={level.id} value={level_index}>
                        {get_level_number(level_index)} — {level.title}
                    </option>
                ))}
            </select>
        </div>
    )
}

function LevelHeader({ level, level_index, sentence_position, completed_count }: {
    level: QuizLevel
    level_index: number
    sentence_position: number
    completed_count: number
}) {
    const level_progress = level.sentences.length === 0 ? 0 : (completed_count / level.sentences.length) * 100

    return (
        <div className="mb-5 rounded-[1.5rem] border border-slate-200/80 bg-white px-5 py-4 shadow-[0_20px_60px_-48px_rgba(15,23,42,0.3)] sm:px-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                    <span className="rounded-full bg-violet-100 px-3 py-1.5 text-xs font-bold uppercase tracking-[0.14em] text-violet-700">
                        Level {get_level_number(level_index)}
                    </span>
                    <div>
                        <h1 className="text-base font-semibold text-slate-950">{level.title}</h1>
                        <p className="text-xs text-slate-500">{level.description}</p>
                    </div>
                </div>
                <p className="text-sm font-medium text-slate-500">
                    Sentence <span className="font-semibold text-slate-900">{sentence_position}</span> / {level.sentences.length}
                </p>
            </div>

            <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-slate-100">
                <div className="h-full rounded-full bg-gradient-to-r from-violet-600 to-cyan-500 transition-all duration-500" style={{ width: `${level_progress}%` }} />
            </div>
        </div>
    )
}

function ChunkProgress({ current_chunk_index, total_chunks }: { current_chunk_index: number; total_chunks: number }) {
    return (
        <div className="flex items-center justify-between gap-5">
            <div>
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Sentence progress</p>
                <p className="mt-1 text-sm font-semibold text-slate-700">
                    Chunk {Math.min(current_chunk_index + 1, total_chunks)} of {total_chunks}
                </p>
            </div>
            <div className="flex flex-1 items-center gap-1.5 sm:max-w-sm">
                {Array.from({ length: total_chunks }).map((_, index) => (
                    <span
                        className={`h-2 flex-1 rounded-full transition-all duration-300 ${
                            index < current_chunk_index
                                ? 'bg-violet-600'
                                : index === current_chunk_index
                                    ? 'bg-violet-300'
                                    : 'bg-slate-100'
                        }`}
                        key={index}
                    />
                ))}
            </div>
        </div>
    )
}

function SentenceBuilder({ sentence, current_chunk_index }: { sentence: QuizSentence; current_chunk_index: number }) {
    const completed_chunks = sentence.chunks.slice(0, current_chunk_index)

    return (
        <div className="mt-8">
            <div className="flex min-h-24 flex-wrap items-center justify-center gap-2.5 rounded-[1.75rem] border border-slate-200 bg-slate-50/80 px-5 py-6 sm:px-8">
                {completed_chunks.length === 0 && (
                    <span className="text-sm font-medium text-slate-400">Choose the first Japanese chunk</span>
                )}

                {completed_chunks.map((chunk, index) => (
                    <div className="animate-chunk-in rounded-2xl border border-violet-200 bg-white px-4 py-3 shadow-sm" key={`${chunk.japanese}-${index}`}>
                        <p className="font-japanese text-2xl font-semibold text-slate-950">{chunk.japanese}</p>
                    </div>
                ))}

                {current_chunk_index < sentence.chunks.length && completed_chunks.length > 0 && (
                    <div className="flex h-14 min-w-16 animate-pulse items-center justify-center rounded-2xl border-2 border-dashed border-violet-200 bg-violet-50/40 px-4">
                        <span className="h-2 w-2 rounded-full bg-violet-400" />
                    </div>
                )}
            </div>

            <div className="mt-4 flex min-h-20 flex-wrap justify-center gap-2.5">
                {completed_chunks.map((chunk, index) => (
                    <div className="min-w-24 rounded-xl px-3 py-2 text-center" key={`${chunk.english}-${index}`}>
                        <p className="font-japanese text-sm font-semibold text-slate-600">{chunk.japanese}</p>
                        <p className="mt-0.5 text-xs font-medium text-slate-400">{chunk.english}</p>
                    </div>
                ))}
            </div>
        </div>
    )
}

function OptionGrid({ options, feedback_japanese, feedback_state, is_locked, on_select_option }: {
    options: QuizOption[]
    feedback_japanese: string
    feedback_state: 'correct' | 'wrong' | null
    is_locked: boolean
    on_select_option: (option: QuizOption) => void
}) {
    return (
        <div className="mt-5 grid gap-3 sm:grid-cols-2">
            {options.map((option, index) => {
                const is_feedback_option = feedback_japanese === option.japanese
                const feedback_class = is_feedback_option && feedback_state === 'wrong'
                    ? 'border-rose-300 bg-rose-50 text-rose-700 ring-4 ring-rose-100'
                    : is_feedback_option && feedback_state === 'correct'
                        ? 'border-emerald-300 bg-emerald-50 text-emerald-700 ring-4 ring-emerald-100'
                        : 'border-slate-200 bg-white text-slate-950 hover:-translate-y-0.5 hover:border-violet-300 hover:shadow-lg hover:shadow-violet-100/60'

                return (
                    <button
                        className={`group relative flex min-h-24 items-center justify-center rounded-2xl border px-6 py-5 transition-all duration-150 disabled:cursor-default ${feedback_class}`}
                        disabled={is_locked}
                        key={`${option.japanese}-${index}`}
                        onClick={() => on_select_option(option)}
                        type="button"
                    >
                        <span className="absolute left-3 top-3 flex h-6 w-6 items-center justify-center rounded-lg border border-slate-200 bg-slate-50 text-[11px] font-bold text-slate-400 group-hover:border-violet-200 group-hover:bg-violet-50 group-hover:text-violet-600">
                            {index + 1}
                        </span>
                        <span className="font-japanese text-3xl font-semibold tracking-wide">{option.japanese}</span>
                        {is_feedback_option && feedback_state === 'correct' && <Check className="absolute right-4 top-4" size={19} />}
                        {is_feedback_option && feedback_state === 'wrong' && <X className="absolute right-4 top-4" size={19} />}
                    </button>
                )
            })}
        </div>
    )
}

function StatusToast({ feedback_state }: { feedback_state: 'correct' | 'wrong' | 'complete' | null }) {
    if (!feedback_state) {
        return null
    }

    const status_content = feedback_state === 'wrong'
        ? { icon: <RotateCcw size={17} />, text: 'Sequence reset — rebuild from the beginning.', class_name: 'border-rose-200 bg-rose-50 text-rose-700' }
        : feedback_state === 'complete'
            ? { icon: <Trophy size={17} />, text: 'Perfect run complete.', class_name: 'border-amber-200 bg-amber-50 text-amber-700' }
            : { icon: <Check size={17} />, text: 'Correct chunk.', class_name: 'border-emerald-200 bg-emerald-50 text-emerald-700' }

    return (
        <div className={`mt-4 flex items-center justify-center gap-2 rounded-xl border px-4 py-2.5 text-sm font-semibold ${status_content.class_name}`}>
            {status_content.icon}
            {status_content.text}
        </div>
    )
}

function LevelComplete({ level, on_restart, on_next_level, has_next_level }: {
    level: QuizLevel
    on_restart: () => void
    on_next_level: () => void
    has_next_level: boolean
}) {
    return (
        <section className="rounded-[2rem] border border-slate-200/80 bg-white px-6 py-12 text-center shadow-[0_30px_90px_-48px_rgba(15,23,42,0.35)] sm:px-12 sm:py-16">
            <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-[1.75rem] bg-gradient-to-br from-amber-300 to-orange-500 text-white shadow-xl shadow-orange-200">
                <Trophy size={36} />
            </div>
            <p className="mt-7 text-xs font-bold uppercase tracking-[0.2em] text-orange-600">Level complete</p>
            <h2 className="mt-2 text-4xl font-semibold tracking-[-0.04em] text-slate-950">{level.title}</h2>
            <p className="mx-auto mt-4 max-w-lg leading-7 text-slate-600">
                Every sentence was completed as a perfect run. Your progress has been saved in this browser.
            </p>
            <div className="mt-8 flex flex-col justify-center gap-3 sm:flex-row">
                <button className="inline-flex items-center justify-center gap-2 rounded-xl border border-slate-200 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50" onClick={on_restart} type="button">
                    <RotateCcw size={17} />
                    Practise again
                </button>
                {has_next_level && (
                    <button className="inline-flex items-center justify-center gap-2 rounded-xl bg-slate-950 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-slate-950/20 transition hover:bg-slate-800" onClick={on_next_level} type="button">
                        Next level
                        <ArrowRight size={17} />
                    </button>
                )}
            </div>
        </section>
    )
}

function QuizCard({ sentence, current_chunk_index, options, feedback_japanese, feedback_state, is_locked, on_select_option, on_restart }: {
    sentence: QuizSentence
    current_chunk_index: number
    options: QuizOption[]
    feedback_japanese: string
    feedback_state: 'correct' | 'wrong' | 'complete' | null
    is_locked: boolean
    on_select_option: (option: QuizOption) => void
    on_restart: () => void
}) {
    return (
        <section className="rounded-[2rem] border border-slate-200/80 bg-white p-5 shadow-[0_30px_90px_-48px_rgba(15,23,42,0.35)] sm:p-7 lg:p-8">
            <div className="flex items-start justify-between gap-4">
                <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-violet-700">Build this sentence</p>
                    <h2 className="mt-2 text-2xl font-semibold tracking-[-0.03em] text-slate-950 sm:text-3xl">{sentence.english}</h2>
                </div>
                <button className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-slate-200 text-slate-400 transition hover:bg-slate-50 hover:text-slate-700 disabled:cursor-default disabled:opacity-40" disabled={is_locked} onClick={on_restart} title="Restart sentence" type="button">
                    <RotateCcw size={17} />
                </button>
            </div>

            <div className="mt-7 border-t border-slate-100 pt-6">
                <ChunkProgress current_chunk_index={current_chunk_index} total_chunks={sentence.chunks.length} />
                <SentenceBuilder current_chunk_index={current_chunk_index} sentence={sentence} />
            </div>

            <div className="mt-3 rounded-[1.5rem] bg-slate-50 p-4 sm:p-5">
                <div className="flex items-center justify-between gap-4">
                    <div>
                        <p className="text-sm font-semibold text-slate-800">Choose the next chunk</p>
                        <p className="mt-0.5 text-xs text-slate-500">Japanese only — use keys 1–4 for speed.</p>
                    </div>
                    <div className="hidden items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-400 sm:flex">
                        <Keyboard size={14} />
                        1–4
                    </div>
                </div>

                <OptionGrid
                    feedback_japanese={feedback_japanese}
                    feedback_state={feedback_state === 'complete' ? 'correct' : feedback_state}
                    is_locked={is_locked}
                    on_select_option={on_select_option}
                    options={options}
                />
                <StatusToast feedback_state={feedback_state} />
            </div>
        </section>
    )
}

function App() {
    const [progress, set_progress] = useState<ProgressState>(get_saved_progress)
    const [active_level_index, set_active_level_index] = useState(0)
    const [sentence_order, set_sentence_order] = useState<number[]>(() => get_sentence_order(levels[0], progress.completed_sentence_ids))
    const [sentence_order_index, set_sentence_order_index] = useState(0)
    const [current_chunk_index, set_current_chunk_index] = useState(0)
    const [options, set_options] = useState<QuizOption[]>([])
    const [feedback_japanese, set_feedback_japanese] = useState('')
    const [feedback_state, set_feedback_state] = useState<'correct' | 'wrong' | 'complete' | null>(null)
    const [is_locked, set_is_locked] = useState(false)
    const [option_seed, set_option_seed] = useState(0)
    const [streak, set_streak] = useState(0)
    const feedback_timeout = useRef<number | null>(null)

    const active_level = levels[active_level_index]
    const sentence_index = sentence_order[sentence_order_index] ?? 0
    const active_sentence = active_level?.sentences[sentence_index]
    const completed_count = active_level ? get_completed_count(active_level, progress.completed_sentence_ids) : 0
    const is_level_complete = Boolean(active_level && active_level.sentences.length > 0 && completed_count === active_level.sentences.length)

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

    const handle_select_option = (option: QuizOption) => {
        if (!active_sentence || is_locked) {
            return
        }

        set_is_locked(true)
        set_feedback_japanese(option.japanese)

        if (!option.is_correct) {
            set_feedback_state('wrong')
            set_streak(0)

            feedback_timeout.current = window.setTimeout(() => {
                set_current_chunk_index(0)
                set_option_seed((current_seed) => current_seed + 1)
                set_feedback_japanese('')
                set_feedback_state(null)
                set_is_locked(false)
            }, 650)
            return
        }

        const next_chunk_index = current_chunk_index + 1
        const sentence_is_complete = next_chunk_index === active_sentence.chunks.length

        set_streak((current_streak) => current_streak + 1)
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
            }, 900)
            return
        }

        feedback_timeout.current = window.setTimeout(() => {
            set_feedback_japanese('')
            set_feedback_state(null)
            set_is_locked(false)
        }, 260)
    }

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
        set_streak(0)
    }

    const handle_select_level = (level_index: number) => {
        if (feedback_timeout.current !== null) {
            window.clearTimeout(feedback_timeout.current)
            feedback_timeout.current = null
        }

        set_active_level_index(level_index)
        set_sentence_order(get_sentence_order(levels[level_index], progress.completed_sentence_ids))
        set_sentence_order_index(0)
        set_current_chunk_index(0)
        set_option_seed((current_seed) => current_seed + 1)
        set_feedback_state(null)
        set_feedback_japanese('')
        set_is_locked(false)
    }

    const handle_restart_level = () => {
        if (feedback_timeout.current !== null) {
            window.clearTimeout(feedback_timeout.current)
            feedback_timeout.current = null
        }

        if (!active_level) {
            return
        }

        const level_sentence_ids = active_level.sentences.map((sentence) => sentence.id)
        set_progress((current_progress) => ({
            completed_sentence_ids: current_progress.completed_sentence_ids.filter(
                (sentence_id) => !level_sentence_ids.includes(sentence_id),
            ),
        }))
        set_sentence_order(shuffle_items(active_level.sentences.map((_, index) => index)))
        set_sentence_order_index(0)
        set_current_chunk_index(0)
        set_option_seed((current_seed) => current_seed + 1)
    }

    const handle_next_level = () => {
        handle_select_level(Math.min(active_level_index + 1, levels.length - 1))
    }

    const completion_percentage = useMemo(() => {
        const total_sentences = levels.reduce((total, level) => total + level.sentences.length, 0)

        if (total_sentences === 0) {
            return 0
        }

        return Math.round((progress.completed_sentence_ids.length / total_sentences) * 100)
    }, [progress.completed_sentence_ids])

    if (levels.length === 0) {
        return (
            <div className="min-h-screen bg-app">
                <AppHeader streak={streak} />
                <EmptyCurriculum />
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-app">
            <AppHeader streak={streak} />

            <main className="mx-auto flex max-w-7xl gap-6 px-5 py-6 sm:px-8 lg:px-10 lg:py-8">
                <LevelRail
                    active_level_index={active_level_index}
                    completed_sentence_ids={progress.completed_sentence_ids}
                    on_select_level={handle_select_level}
                />

                <div className="min-w-0 flex-1">
                    <MobileLevelSelect active_level_index={active_level_index} on_select_level={handle_select_level} />
                    <LevelHeader
                        completed_count={completed_count}
                        level={active_level}
                        level_index={active_level_index}
                        sentence_position={Math.min(completed_count + 1, active_level.sentences.length)}
                    />

                    {is_level_complete ? (
                        <LevelComplete
                            has_next_level={active_level_index < levels.length - 1}
                            level={active_level}
                            on_next_level={handle_next_level}
                            on_restart={handle_restart_level}
                        />
                    ) : active_sentence ? (
                        <QuizCard
                            current_chunk_index={current_chunk_index}
                            feedback_japanese={feedback_japanese}
                            feedback_state={feedback_state}
                            is_locked={is_locked}
                            on_restart={handle_restart_sentence}
                            on_select_option={handle_select_option}
                            options={options}
                            sentence={active_sentence}
                        />
                    ) : null}

                    <div className="mt-5 flex flex-wrap items-center justify-between gap-3 px-2 text-xs text-slate-400">
                        <div className="flex items-center gap-2">
                            <CircleHelp size={14} />
                            One mistake resets the current sentence.
                        </div>
                        <span>{completion_percentage}% curriculum complete</span>
                    </div>
                </div>
            </main>
        </div>
    )
}

export default App
