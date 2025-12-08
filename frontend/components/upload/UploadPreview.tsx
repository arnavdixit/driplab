import Image from 'next/image'

export type PreviewItem = {
  id: string
  previewUrl: string
}

type UploadPreviewProps = {
  items: PreviewItem[]
  onRemove: (id: string) => void
}

export default function UploadPreview({ items, onRemove }: UploadPreviewProps) {
  if (!items.length) {
    return (
      <div className="rounded-md border border-dashed border-gray-200 bg-white p-6 text-center text-sm text-gray-500">
        No files selected yet.
      </div>
    )
  }

  return (
    <div>
      <h2 className="mb-3 text-lg font-semibold text-gray-900">Preview</h2>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
        {items.map((item) => (
          <div key={item.id} className="relative overflow-hidden rounded-md border border-gray-200 bg-white shadow-sm">
            <Image
              src={item.previewUrl}
              alt="Selected preview"
              width={400}
              height={300}
              className="h-36 w-full object-cover"
            />
            <button
              type="button"
              onClick={() => onRemove(item.id)}
              className="absolute right-2 top-2 rounded-full bg-black/70 px-2 py-1 text-xs font-semibold text-white hover:bg-black"
              aria-label="Remove image"
            >
              Remove
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}
