import { DragEvent, useCallback, useRef, useState } from 'react'

export type FileError = { fileName: string; reason: string }
export type FileSelectHandler = (files: File[], errors: FileError[]) => void

type DropZoneProps = {
  onSelect: FileSelectHandler
  allowedTypes: string[]
  maxSizeBytes: number
  disabled?: boolean
}

function validateFile(file: File, allowedTypes: string[], maxSizeBytes: number): FileError | null {
  if (!allowedTypes.includes(file.type)) {
    return { fileName: file.name, reason: 'Unsupported file type' }
  }
  if (file.size > maxSizeBytes) {
    return { fileName: file.name, reason: 'File too large (max 10MB)' }
  }
  return null
}

export default function DropZone({ onSelect, allowedTypes, maxSizeBytes, disabled }: DropZoneProps) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const processFiles = useCallback(
    (fileList: FileList | null) => {
      if (!fileList) return
      const files = Array.from(fileList)
      const accepted: File[] = []
      const errors: FileError[] = []

      files.forEach((file) => {
        const error = validateFile(file, allowedTypes, maxSizeBytes)
        if (error) {
          errors.push(error)
        } else {
          accepted.push(file)
        }
      })

      onSelect(accepted, errors)
    },
    [allowedTypes, maxSizeBytes, onSelect],
  )

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    if (disabled) return
    setIsDragging(false)
    processFiles(event.dataTransfer?.files ?? null)
  }

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    if (disabled) return
    setIsDragging(true)
  }

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)
  }

  const handleBrowse = () => {
    if (disabled) return
    inputRef.current?.click()
  }

  const handleInputChange = () => {
    if (disabled) return
    processFiles(inputRef.current?.files ?? null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-14 text-center transition ${
        isDragging ? 'border-indigo-500 bg-indigo-50' : 'border-gray-300 bg-gray-50'
      } ${disabled ? 'cursor-not-allowed opacity-70' : ''}`}
      role="button"
      tabIndex={0}
      onClick={handleBrowse}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          handleBrowse()
        }
      }}
    >
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept={allowedTypes.join(',')}
        multiple
        onChange={handleInputChange}
        aria-label="Upload images"
      />
      <div className="flex flex-col items-center gap-2">
        <span className="rounded-full bg-indigo-100 px-3 py-1 text-sm font-semibold text-indigo-700">Upload</span>
        <p className="text-lg font-semibold text-gray-900">Drag & drop images here</p>
        <p className="text-sm text-gray-600">or click to browse files</p>
        <p className="mt-2 text-xs text-gray-500">
          Supported: JPEG, PNG, WebP · Max size: {Math.round(maxSizeBytes / (1024 * 1024))}MB
        </p>
      </div>
    </div>
  )
}
