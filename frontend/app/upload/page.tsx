'use client'

import { useEffect, useMemo, useState } from 'react'
import DropZone, { FileError, FileSelectHandler } from '@/components/upload/DropZone'
import UploadPreview, { PreviewItem } from '@/components/upload/UploadPreview'
import UploadProgress, { UploadState } from '@/components/upload/UploadProgress'
import { uploadGarment } from '@/lib/upload'

type UploadItem = PreviewItem &
  UploadState & {
    file: File
  }

const allowedTypes = ['image/jpeg', 'image/png', 'image/webp']
const maxSizeBytes = 10 * 1024 * 1024

export default function UploadPage() {
  const [items, setItems] = useState<UploadItem[]>([])
  const [errors, setErrors] = useState<FileError[]>([])
  const [isUploading, setIsUploading] = useState(false)

  const handleSelect: FileSelectHandler = (files, fileErrors) => {
    setErrors(fileErrors)
    if (!files.length) return

    const nextItems = files.map<UploadItem>((file) => ({
      id: crypto.randomUUID(),
      file,
      previewUrl: URL.createObjectURL(file),
      status: 'queued',
      progress: 0,
    }))
    setItems((prev) => [...prev, ...nextItems])
  }

  const handleRemove = (id: string) => {
    setItems((prev) => {
      const match = prev.find((item) => item.id === id)
      if (match) {
        URL.revokeObjectURL(match.previewUrl)
      }
      return prev.filter((item) => item.id !== id)
    })
  }

  const reset = () => {
    setItems([])
    setErrors([])
    setIsUploading(false)
  }

  useEffect(() => {
    return () => {
      items.forEach((item) => URL.revokeObjectURL(item.previewUrl))
    }
  }, [items])

  const startUpload = async () => {
    if (!items.length || isUploading) return
    setIsUploading(true)

    for (const item of items) {
      setItems((prev) =>
        prev.map((p) => (p.id === item.id ? { ...p, status: 'uploading', progress: 0, error: undefined } : p)),
      )
      try {
        await uploadGarment(item.file, {
          onProgress: (progress) =>
            setItems((prev) => prev.map((p) => (p.id === item.id ? { ...p, progress } : p))),
        })
        setItems((prev) => prev.map((p) => (p.id === item.id ? { ...p, status: 'success', progress: 100 } : p)))
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Upload failed'
        setItems((prev) => prev.map((p) => (p.id === item.id ? { ...p, status: 'error', error: message } : p)))
      }
    }

    setIsUploading(false)
  }

  const hasInProgress = useMemo(
    () => items.some((item) => item.status === 'uploading' || item.status === 'queued'),
    [items],
  )

  return (
    <main className="min-h-screen bg-white text-gray-900">
      <section className="mx-auto max-w-5xl px-6 py-12">
        <header className="mb-8 text-center">
          <p className="text-sm font-semibold uppercase tracking-wide text-indigo-600">Upload</p>
          <h1 className="mt-2 text-4xl font-bold">Add your garment photos</h1>
          <p className="mt-3 text-gray-600">
            Drag and drop images or browse files. Supported types: JPEG, PNG, WebP. Max size: 10MB per file.
          </p>
        </header>

        <DropZone
          onSelect={handleSelect}
          allowedTypes={allowedTypes}
          maxSizeBytes={maxSizeBytes}
          disabled={isUploading}
        />

        {errors.length > 0 && (
          <div className="mt-4 rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            <p className="font-semibold">Some files were skipped:</p>
            <ul className="mt-2 list-disc space-y-1 pl-5">
              {errors.map((error) => (
                <li key={`${error.fileName}-${error.reason}`}>
                  {error.fileName}: {error.reason}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="mt-8 space-y-6">
          <UploadPreview items={items} onRemove={handleRemove} />
          <UploadProgress items={items} />
        </div>

        <div className="mt-8 flex items-center gap-3">
          <button
            type="button"
            onClick={startUpload}
            disabled={!items.length || isUploading || !hasInProgress}
            className="rounded-md bg-indigo-600 px-4 py-2 text-white shadow hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-gray-300"
          >
            {isUploading ? 'Uploading…' : 'Start Upload'}
          </button>
          <button
            type="button"
            onClick={reset}
            disabled={!items.length || isUploading}
            className="rounded-md border border-gray-300 px-4 py-2 text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-60"
          >
            Clear
          </button>
        </div>
      </section>
    </main>
  )
}
