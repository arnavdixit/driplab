import { Garment, GarmentStatus } from '@/types/garment'

type GarmentCardProps = {
  garment: Garment
  onClick?: (garmentId: string) => void
}

const statusStyles: Record<GarmentStatus, string> = {
  pending: 'bg-amber-100 text-amber-800 ring-amber-200',
  processing: 'bg-sky-100 text-sky-800 ring-sky-200',
  ready: 'bg-emerald-100 text-emerald-800 ring-emerald-200',
  failed: 'bg-rose-100 text-rose-800 ring-rose-200',
}

const statusLabel: Record<GarmentStatus, string> = {
  pending: 'Pending',
  processing: 'Processing',
  ready: 'Ready',
  failed: 'Failed',
}

const apiBase = process.env.NEXT_PUBLIC_API_URL

const resolveImageUrl = (path?: string | null) => {
  if (!path) return null
  if (path.startsWith('http://') || path.startsWith('https://')) return path
  if (!apiBase) return null
  return `${apiBase}${path.startsWith('/') ? path : `/${path}`}`
}

const extractColors = (garment: Garment): string[] => {
  const attributes = garment.prediction?.attributes
  if (!attributes) return []
  const rawColor = (attributes as Record<string, unknown>).color
  if (typeof rawColor === 'string') return [rawColor]
  if (Array.isArray(rawColor)) return rawColor.map((value) => String(value))
  return []
}

export default function GarmentCard({ garment, onClick }: GarmentCardProps) {
  const imageSrc =
    resolveImageUrl(garment.thumbnail_path) ??
    resolveImageUrl(garment.processed_image_path) ??
    resolveImageUrl(garment.original_image_path)

  const name = garment.custom_name || garment.prediction?.category || 'Untitled garment'
  const category = garment.prediction?.category
  const colors = extractColors(garment)

  const statusClass = statusStyles[garment.status]
  const statusText = statusLabel[garment.status]

  return (
    <article
      className="group h-full cursor-pointer overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm transition hover:-translate-y-0.5 hover:shadow-md"
      onClick={() => onClick?.(garment.id)}
    >
      <div className="aspect-[4/5] w-full overflow-hidden bg-gradient-to-br from-gray-50 to-gray-100">
        {imageSrc ? (
          <img
            src={imageSrc}
            alt={name}
            loading="lazy"
            className="h-full w-full object-cover transition duration-300 group-hover:scale-[1.02]"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center text-sm text-gray-500">No image</div>
        )}
      </div>

      <div className="space-y-2 px-4 py-3">
        <div className="flex items-start justify-between gap-2">
          <h3 className="line-clamp-2 text-base font-semibold text-gray-900">{name}</h3>
          <span
            className={`rounded-full px-2 py-1 text-xs font-medium ring-1 ring-inset ${statusClass}`}
            aria-label={`Status: ${statusText}`}
          >
            {statusText}
          </span>
        </div>

        <div className="flex flex-wrap gap-2">
          {category && <span className="rounded-full bg-indigo-50 px-2 py-1 text-xs font-medium text-indigo-700">{category}</span>}
          {colors.map((color) => (
            <span key={color} className="rounded-full bg-gray-100 px-2 py-1 text-xs font-medium text-gray-700">
              {color}
            </span>
          ))}
          {colors.length === 0 && <span className="text-xs text-gray-500">No colors</span>}
        </div>

        {garment.custom_notes && (
          <p className="line-clamp-2 text-xs text-gray-600" title={garment.custom_notes}>
            {garment.custom_notes}
          </p>
        )}
      </div>
    </article>
  )
}
