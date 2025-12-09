'use client'

import { useMemo, useState } from 'react'

import FilterBar, { FilterValues } from '@/components/wardrobe/FilterBar'
import GarmentGrid from '@/components/wardrobe/GarmentGrid'
import { useWardrobe } from '@/hooks/useWardrobe'
import { Garment } from '@/types/garment'

const initialFilters: FilterValues = {
  search: '',
  category: '',
  color: '',
  status: '',
}

const collectColors = (garments: Garment[]) => {
  const colors = new Set<string>()
  garments.forEach((item) => {
    const attributes = item.prediction?.attributes
    const rawColor = attributes ? (attributes as Record<string, unknown>).color : null
    if (typeof rawColor === 'string') {
      colors.add(rawColor)
    } else if (Array.isArray(rawColor)) {
      rawColor.forEach((value) => colors.add(String(value)))
    }
  })
  return Array.from(colors).sort()
}

const collectCategories = (garments: Garment[]) => {
  const categories = new Set<string>()
  garments.forEach((item) => {
    if (item.prediction?.category) {
      categories.add(item.prediction.category)
    }
  })
  return Array.from(categories).sort()
}

export default function WardrobePage() {
  const [filters, setFilters] = useState<FilterValues>(initialFilters)

  const { items, loading, isLoadingMore, error, hasMore, total, loadMore, refetch } = useWardrobe({
    pageSize: 12,
    filters,
  })

  const categories = useMemo(() => collectCategories(items), [items])
  const colors = useMemo(() => collectColors(items), [items])

  return (
    <main className="min-h-screen bg-gray-50 text-gray-900">
      <section className="mx-auto max-w-6xl px-6 py-12">
        <header className="mb-6 space-y-2">
          <p className="text-sm font-semibold uppercase tracking-wide text-indigo-600">Wardrobe</p>
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <h1 className="text-4xl font-bold">Your wardrobe</h1>
            <span className="text-sm text-gray-600">{total} items</span>
          </div>
          <p className="text-sm text-gray-600">
            Browse all garments, filter by category, color, status, and search by name. Click load more to fetch additional items.
          </p>
        </header>

        <FilterBar values={filters} categories={categories} colors={colors} onChange={setFilters} />

        {error && (
          <div className="mt-4 flex items-center justify-between rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
            <span>{error}</span>
            <button
              type="button"
              onClick={refetch}
              className="rounded-md border border-rose-300 bg-white px-3 py-1 text-rose-800 hover:bg-rose-100"
            >
              Retry
            </button>
          </div>
        )}

        <div className="mt-6">
          {loading && items.length === 0 ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {Array.from({ length: 8 }).map((_, index) => (
                <div
                  key={index}
                  className="h-full overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm"
                  aria-label="Loading garment card"
                >
                  <div className="aspect-[4/5] w-full animate-pulse bg-gray-200" />
                  <div className="space-y-3 p-4">
                    <div className="h-4 w-2/3 animate-pulse rounded bg-gray-200" />
                    <div className="flex gap-2">
                      <div className="h-5 w-16 animate-pulse rounded-full bg-gray-200" />
                      <div className="h-5 w-12 animate-pulse rounded-full bg-gray-200" />
                    </div>
                    <div className="h-3 w-full animate-pulse rounded bg-gray-200" />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <GarmentGrid garments={items} emptyMessage="No garments yet. Upload some photos to populate your wardrobe." />
          )}
        </div>

        <div className="mt-8 flex items-center justify-center gap-3">
          {hasMore && (
            <button
              type="button"
              onClick={loadMore}
              disabled={isLoadingMore || loading}
              className="rounded-md bg-indigo-600 px-4 py-2 text-white shadow hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-indigo-300"
            >
              {isLoadingMore ? 'Loading…' : 'Load more'}
            </button>
          )}
          {!hasMore && items.length > 0 && <span className="text-sm text-gray-600">You are up to date.</span>}
        </div>
      </section>
    </main>
  )
}
