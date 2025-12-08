export type UploadState = {
  id: string
  status: 'queued' | 'uploading' | 'success' | 'error'
  progress: number
  error?: string
}

type UploadProgressProps = {
  items: UploadState[]
}

const statusLabel: Record<UploadState['status'], string> = {
  queued: 'Queued',
  uploading: 'Uploading',
  success: 'Uploaded',
  error: 'Error',
}

export default function UploadProgress({ items }: UploadProgressProps) {
  if (!items.length) {
    return null
  }

  return (
    <div>
      <h2 className="mb-3 text-lg font-semibold text-gray-900">Progress</h2>
      <ul className="space-y-3">
        {items.map((item) => (
          <li key={item.id} className="rounded-md border border-gray-200 bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-800">{statusLabel[item.status]}</span>
              <span className="text-xs text-gray-500">{item.progress}%</span>
            </div>
            <div className="mt-2 h-2 w-full rounded-full bg-gray-100">
              <div
                className={`h-2 rounded-full ${
                  item.status === 'error'
                    ? 'bg-red-500'
                    : item.status === 'success'
                      ? 'bg-green-500'
                      : 'bg-indigo-500'
                }`}
                style={{ width: `${Math.min(item.progress, 100)}%` }}
              />
            </div>
            {item.error && <p className="mt-2 text-xs text-red-600">{item.error}</p>}
          </li>
        ))}
      </ul>
    </div>
  )
}
