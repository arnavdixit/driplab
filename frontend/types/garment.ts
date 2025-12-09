export type GarmentStatus = 'pending' | 'processing' | 'ready' | 'failed'

export type GarmentPrediction = {
  category: string
  subcategory?: string | null
  attributes?: Record<string, unknown> | null
  bbox_x?: number | null
  bbox_y?: number | null
  bbox_width?: number | null
  bbox_height?: number | null
  detection_confidence?: number | null
  category_confidence?: number | null
  embedding_id?: string | null
}

export type Garment = {
  id: string
  status: GarmentStatus
  original_image_path: string
  processed_image_path?: string | null
  thumbnail_path?: string | null
  error_message?: string | null
  custom_name?: string | null
  custom_notes?: string | null
  created_at: string
  updated_at: string
  prediction?: GarmentPrediction | null
}

export type WardrobeListResponse = {
  items: Garment[]
  total: number
  limit: number
  offset: number
}

export type WardrobeFilters = {
  search?: string
  category?: string
  color?: string
  status?: GarmentStatus | ''
}
