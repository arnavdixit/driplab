type UploadOptions = {
  onProgress?: (progress: number) => void
}

const apiBase = process.env.NEXT_PUBLIC_API_URL

export async function uploadGarment(file: File, options?: UploadOptions): Promise<void> {
  if (!apiBase) {
    throw new Error('NEXT_PUBLIC_API_URL is not set')
  }

  const endpoint = `${apiBase}/api/v1/wardrobe/upload`
  const formData = new FormData()
  formData.append('file', file)

  await new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest()

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && options?.onProgress) {
        const percent = Math.round((event.loaded / event.total) * 100)
        options.onProgress(percent)
      }
    }

    xhr.onerror = () => reject(new Error('Network error during upload'))
    xhr.ontimeout = () => reject(new Error('Upload timed out'))
    xhr.onreadystatechange = () => {
      if (xhr.readyState === XMLHttpRequest.DONE) {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve()
        } else {
          const message = xhr.responseText || 'Upload failed'
          reject(new Error(message))
        }
      }
    }

    xhr.open('POST', endpoint)
    xhr.send(formData)
  })
}
