'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Garment, GarmentStatus, WardrobeFilters, WardrobeListResponse } from '@/types/garment'

type UseWardrobeOptions = {
  pageSize?: number
  filters?: WardrobeFilters
}

type UseWardrobeResult = {
  items: Garment[]
  loading: boolean
  isLoadingMore: boolean
  error: string | null
  hasMore: boolean
  total: number
  loadMore: () => Promise<void>
  refetch: () => Promise<void>
}

const apiBase = process.env.NEXT_PUBLIC_API_URL

const matchesClientFilters = (item: Garment, filters: WardrobeFilters): boolean => {
  const { search, category, color, status } = filters
  const lcSearch = search?.trim().toLowerCase()
  const hasSearch = lcSearch?.length
  const itemName = item.custom_name ?? item.prediction?.category ?? ''
  const matchesSearch = hasSearch ? itemName.toLowerCase().includes(lcSearch) : true

  const matchesCategory = category ? item.prediction?.category === category : true
  const matchesStatus = status ? item.status === status : true

  if (color) {
    const attributes = item.prediction?.attributes
    const colors =
      typeof attributes?.color === 'string'
        ? [attributes.color]
        : Array.isArray(attributes?.color)
          ? attributes.color
          : []
    const normalized = colors.map((c) => String(c).toLowerCase())
    if (normalized.length === 0) {
      return false
    }
    return matchesSearch && matchesCategory && matchesStatus && normalized.includes(color.toLowerCase())
  }

  return matchesSearch && matchesCategory && matchesStatus
}

const uniqueMerge = (existing: Garment[], incoming: Garment[]) => {
  const map = new Map<string, Garment>()
  for (const item of existing) {
    map.set(item.id, item)
  }
  for (const item of incoming) {
    map.set(item.id, item)
  }
  return Array.from(map.values())
}

export function useWardrobe(options: UseWardrobeOptions = {}): UseWardrobeResult {
  const { pageSize = 12, filters = {} } = options

  const [items, setItems] = useState<Garment[]>([])
  const [loading, setLoading] = useState(false)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hasMore, setHasMore] = useState(true)
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(0)

  const filtersRef = useRef<WardrobeFilters>(filters)

  const filterSignature = useMemo(() => JSON.stringify(filters), [filters])

  const fetchPage = useCallback(
    async (pageIndex: number, append: boolean) => {
      if (!apiBase) {
        setError('NEXT_PUBLIC_API_URL is not set')
        setLoading(false)
        setIsLoadingMore(false)
        return
      }

      const isFirstPage = pageIndex === 0 && !append
      if (isFirstPage) {
        setLoading(true)
      } else {
        setIsLoadingMore(true)
      }
      setError(null)

      const params = new URLSearchParams()
      params.set('limit', String(pageSize))
      params.set('offset', String(pageIndex * pageSize))

      const activeFilters = filtersRef.current
      if (activeFilters.status) params.set('status', activeFilters.status)
      if (activeFilters.category) params.set('category', activeFilters.category)
      if (activeFilters.search) params.set('search', activeFilters.search)
      if (activeFilters.color) params.set('color', activeFilters.color)

      try {
        const response = await fetch(`${apiBase}/api/v1/wardrobe?${params.toString()}`, {
          cache: 'no-store',
        })
        if (!response.ok) {
          const message = await response.text()
          throw new Error(message || 'Failed to fetch wardrobe')
        }

        const data = (await response.json()) as WardrobeListResponse
        const filteredItems = data.items.filter((item) => matchesClientFilters(item, activeFilters))

        setItems((prev) => (append ? uniqueMerge(prev, filteredItems) : filteredItems))
        setTotal(data.total)
        setHasMore(data.offset + data.limit < data.total)
        setPage(pageIndex)
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unexpected error loading wardrobe'
        setError(message)
      } finally {
        setLoading(false)
        setIsLoadingMore(false)
      }
    },
    [pageSize],
  )

  useEffect(() => {
    filtersRef.current = filters
  }, [filterSignature, filters])

  useEffect(() => {
    fetchPage(0, false)
  }, [filterSignature, fetchPage])

  const loadMore = useCallback(async () => {
    if (loading || isLoadingMore || !hasMore) return
    const nextPage = page + 1
    await fetchPage(nextPage, true)
  }, [fetchPage, hasMore, isLoadingMore, loading, page])

  const refetch = useCallback(async () => {
    await fetchPage(0, false)
  }, [fetchPage])

  return {
    items,
    loading,
    isLoadingMore,
    error,
    hasMore,
    total,
    loadMore,
    refetch,
  }
}
