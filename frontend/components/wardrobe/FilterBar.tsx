'use client'

import { ChangeEvent, useEffect, useMemo, useState } from 'react'

import { GarmentStatus } from '@/types/garment'

export type FilterValues = {
  search: string
  category: string
  color: string
  status: GarmentStatus | ''
}

type FilterBarProps = {
  values: FilterValues
  categories: string[]
  colors: string[]
  onChange: (next: FilterValues) => void
}

const statusOptions: Array<{ value: GarmentStatus | ''; label: string }> = [
  { value: '', label: 'All statuses' },
  { value: 'pending', label: 'Pending' },
  { value: 'processing', label: 'Processing' },
  { value: 'ready', label: 'Ready' },
  { value: 'failed', label: 'Failed' },
]

export default function FilterBar({ values, categories, colors, onChange }: FilterBarProps) {
  const [searchInput, setSearchInput] = useState(values.search)

  useEffect(() => {
    setSearchInput(values.search)
  }, [values.search])

  useEffect(() => {
    const handle = setTimeout(() => {
      if (searchInput === values.search) return
      onChange({ ...values, search: searchInput })
    }, 300)
    return () => clearTimeout(handle)
  }, [onChange, searchInput, values])

  const categoryOptions = useMemo(() => ['All categories', ...categories], [categories])
  const colorOptions = useMemo(() => ['All colors', ...colors], [colors])

  const handleSelect = (key: keyof FilterValues) => (event: ChangeEvent<HTMLSelectElement>) => {
    onChange({ ...values, [key]: event.target.value })
  }

  const handleStatusChange = (event: ChangeEvent<HTMLSelectElement>) => {
    onChange({ ...values, status: event.target.value as GarmentStatus | '' })
  }

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-gray-200 bg-white p-4 shadow-sm lg:flex-row lg:items-end lg:justify-between">
      <div className="flex flex-1 flex-wrap items-center gap-3">
        <div className="flex-1 min-w-[220px]">
          <label htmlFor="wardrobe-search" className="sr-only">
            Search garments
          </label>
          <input
            id="wardrobe-search"
            type="search"
            value={searchInput}
            placeholder="Search by name"
            onChange={(event) => setSearchInput(event.target.value)}
            className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
          />
        </div>

        <div className="min-w-[160px]">
          <label htmlFor="wardrobe-category" className="sr-only">
            Category
          </label>
          <select
            id="wardrobe-category"
            value={values.category}
            onChange={handleSelect('category')}
            className="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
          >
            {categoryOptions.map((option) => (
              <option key={option} value={option === 'All categories' ? '' : option}>
                {option}
              </option>
            ))}
          </select>
        </div>

        <div className="min-w-[160px]">
          <label htmlFor="wardrobe-color" className="sr-only">
            Color
          </label>
          <select
            id="wardrobe-color"
            value={values.color}
            onChange={handleSelect('color')}
            className="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
          >
            {colorOptions.map((option) => (
              <option key={option} value={option === 'All colors' ? '' : option}>
                {option}
              </option>
            ))}
          </select>
        </div>

        <div className="min-w-[160px]">
          <label htmlFor="wardrobe-status" className="sr-only">
            Status
          </label>
          <select
            id="wardrobe-status"
            value={values.status}
            onChange={handleStatusChange}
            className="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
          >
            {statusOptions.map((option) => (
              <option key={option.value || 'all'} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  )
}
