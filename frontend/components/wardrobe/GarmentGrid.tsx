import GarmentCard from './GarmentCard'

import { Garment } from '@/types/garment'

type GarmentGridProps = {
  garments: Garment[]
  onSelect?: (garmentId: string) => void
  emptyMessage?: string
}

export default function GarmentGrid({ garments, onSelect, emptyMessage }: GarmentGridProps) {
  if (!garments.length) {
    return (
      <div className="rounded-lg border border-dashed border-gray-200 bg-gray-50 px-4 py-10 text-center text-sm text-gray-600">
        {emptyMessage || 'No garments yet. Upload to get started.'}
      </div>
    )
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {garments.map((garment) => (
        <GarmentCard key={garment.id} garment={garment} onClick={onSelect} />
      ))}
    </div>
  )
}
