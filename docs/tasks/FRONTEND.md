# Frontend Tasks (FE-xxx)

All frontend tasks for Next.js UI. **Built by Cursor, reviewed by user.**

---

## FE-001: Next.js Project Setup

**Phase:** 1 | **Priority:** P0 | **Dependencies:** None

**Description:**
Initialize Next.js project with TypeScript, Tailwind CSS, and App Router.

**Files to Create:**
- `frontend/package.json` - Dependencies
- `frontend/tsconfig.json` - TypeScript config
- `frontend/tailwind.config.ts` - Tailwind config
- `frontend/src/app/layout.tsx` - Root layout
- `frontend/src/app/page.tsx` - Home page
- `frontend/src/lib/api.ts` - API client

**Key Requirements:**
- Next.js 14+ with App Router
- TypeScript strict mode
- Tailwind CSS with custom theme
- API client configured for backend URL
- Dark/light theme support (optional)

**Acceptance Criteria:**
- [ ] `npm run dev` starts dev server
- [ ] Home page renders
- [ ] Tailwind styles work
- [ ] API client can reach backend

---

## FE-002: Upload Interface (Drag-Drop)

**Phase:** 1 | **Priority:** P0 | **Dependencies:** BE-004

**Description:**
Create drag-and-drop interface for uploading garment photos.

**Files to Create:**
- `frontend/src/app/upload/page.tsx` - Upload page
- `frontend/src/components/upload/DropZone.tsx` - Drag-drop zone
- `frontend/src/components/upload/UploadProgress.tsx` - Progress indicator
- `frontend/src/components/upload/UploadPreview.tsx` - Image preview

**Key Requirements:**
- Drag-and-drop file selection
- Click to browse files
- Multiple file upload support
- Image preview before upload
- Progress indicator during upload
- Success/error states
- File type and size validation (client-side)

**Acceptance Criteria:**
- [ ] Can drag files onto zone
- [ ] Can click to browse
- [ ] Shows preview of selected images
- [ ] Uploads to backend successfully
- [ ] Shows progress during upload
- [ ] Shows error for invalid files

---

## FE-003: Wardrobe Gallery View

**Phase:** 1 | **Priority:** P0 | **Dependencies:** BE-006

**Description:**
Grid view of all wardrobe items with filtering and search.

**Files to Create:**
- `frontend/src/app/wardrobe/page.tsx` - Wardrobe page
- `frontend/src/components/wardrobe/GarmentGrid.tsx` - Grid layout
- `frontend/src/components/wardrobe/GarmentCard.tsx` - Item card
- `frontend/src/components/wardrobe/FilterBar.tsx` - Filter controls
- `frontend/src/hooks/useWardrobe.ts` - Data fetching hook
- `frontend/src/types/garment.ts` - TypeScript types

**Key Requirements:**
- Responsive grid layout
- Thumbnail images with lazy loading
- Category and color badges
- Processing status indicator
- Filter by category, color, status
- Search by name (if implemented)
- Pagination or infinite scroll

**Acceptance Criteria:**
- [ ] Shows all wardrobe items
- [ ] Responsive grid (2-4 columns)
- [ ] Can filter by category
- [ ] Shows processing status
- [ ] Loads more on scroll/click

---

## FE-004: Garment Detail View

**Phase:** 2 | **Priority:** P1 | **Dependencies:** BE-006

**Description:**
Detail page for single garment with full attributes and edit capability.

**Files to Create:**
- `frontend/src/app/wardrobe/[id]/page.tsx` - Detail page
- `frontend/src/components/wardrobe/GarmentDetail.tsx` - Detail component
- `frontend/src/components/wardrobe/AttributeEditor.tsx` - Edit attributes

**Key Requirements:**
- Large image view
- All predicted attributes displayed
- Edit custom name and notes
- Correct misclassified attributes
- Delete garment button
- Back to gallery navigation

**Acceptance Criteria:**
- [ ] Shows full-size image
- [ ] Displays all attributes
- [ ] Can edit name/notes
- [ ] Can delete item
- [ ] Navigates back to gallery

---

## FE-005: Processing Status Indicator

**Phase:** 2 | **Priority:** P1 | **Dependencies:** BE-008

**Description:**
Show real-time processing status for uploaded garments.

**Files to Modify:**
- `frontend/src/components/wardrobe/GarmentCard.tsx` - Add status
- `frontend/src/hooks/useWardrobe.ts` - Add polling

**States:**
- `pending` - Waiting in queue
- `processing` - ML pipeline running
- `ready` - Processing complete
- `failed` - Error occurred

**Key Requirements:**
- Visual indicator for each state
- Poll for updates while pending/processing
- Show error message if failed
- Stop polling when ready

**Acceptance Criteria:**
- [ ] Shows spinner for pending/processing
- [ ] Shows checkmark for ready
- [ ] Shows error icon for failed
- [ ] Updates automatically

---

## FE-006: Outfit Recommendation Cards

**Phase:** 3 | **Priority:** P0 | **Dependencies:** BE-009

**Description:**
Display outfit recommendations as cards with garment images.

**Files to Create:**
- `frontend/src/app/outfits/page.tsx` - Outfits page
- `frontend/src/components/outfits/OutfitCard.tsx` - Outfit card
- `frontend/src/components/outfits/OutfitGrid.tsx` - Grid layout
- `frontend/src/components/outfits/OccasionSelector.tsx` - Occasion picker
- `frontend/src/hooks/useOutfits.ts` - Data fetching hook
- `frontend/src/types/outfit.ts` - TypeScript types

**Key Requirements:**
- Outfit card shows all garment images
- Compatibility score displayed
- Explanation text shown
- Occasion/context selector
- Request new recommendations button

**Acceptance Criteria:**
- [ ] Shows outfit as image grid
- [ ] Displays score and explanation
- [ ] Can select occasion
- [ ] Can request recommendations

---

## FE-007: Like/Dislike Buttons

**Phase:** 3 | **Priority:** P1 | **Dependencies:** BE-011

**Description:**
Feedback buttons on outfit cards to like or dislike recommendations.

**Files to Modify:**
- `frontend/src/components/outfits/OutfitCard.tsx` - Add buttons
- `frontend/src/components/outfits/FeedbackButtons.tsx` - Button component

**Actions:**
- Like (thumbs up)
- Dislike (thumbs down)
- Save (bookmark)
- "I wore this" (checkmark)

**Key Requirements:**
- Clear visual feedback on click
- Optimistic UI update
- Send feedback to backend
- Prevent duplicate feedback

**Acceptance Criteria:**
- [ ] Buttons visible on cards
- [ ] Visual state changes on click
- [ ] Feedback sent to backend
- [ ] Handles errors gracefully

---

## FE-008: Outfit Detail Modal

**Phase:** 3 | **Priority:** P1 | **Dependencies:** FE-006

**Description:**
Modal/drawer showing outfit details when clicking a card.

**Files to Create:**
- `frontend/src/components/outfits/OutfitDetail.tsx` - Detail modal

**Key Requirements:**
- Larger garment images
- Detailed explanation
- Score breakdown (color, formality, etc.)
- Links to individual garments
- Feedback buttons
- Close button

**Acceptance Criteria:**
- [ ] Opens on card click
- [ ] Shows larger images
- [ ] Shows score breakdown
- [ ] Can navigate to garments
- [ ] Closes properly

---

## FE-009: Chat Interface

**Phase:** 4 | **Priority:** P0 | **Dependencies:** BE-018

**Description:**
Chat interface for natural language outfit requests.

**Files to Create:**
- `frontend/src/app/chat/page.tsx` - Chat page
- `frontend/src/components/chat/ChatContainer.tsx` - Main container
- `frontend/src/components/chat/ChatMessage.tsx` - Message bubble
- `frontend/src/components/chat/ChatInput.tsx` - Input field
- `frontend/src/hooks/useChat.ts` - Chat logic hook
- `frontend/src/types/chat.ts` - TypeScript types

**Key Requirements:**
- Message history display
- User and assistant message styling
- Text input with send button
- Loading indicator while waiting
- Auto-scroll to latest message
- Conversation persistence

**Acceptance Criteria:**
- [ ] Can send messages
- [ ] Shows message history
- [ ] Displays loading state
- [ ] Auto-scrolls on new messages

---

## FE-010: Streaming Response Display

**Phase:** 4 | **Priority:** P0 | **Dependencies:** FE-009

**Description:**
Display streaming responses from chat endpoint.

**Files to Modify:**
- `frontend/src/hooks/useChat.ts` - Add streaming support
- `frontend/src/components/chat/ChatMessage.tsx` - Streaming render

**Key Requirements:**
- Connect to SSE endpoint
- Display text as it streams
- Show typing indicator
- Handle stream completion
- Handle stream errors

**Acceptance Criteria:**
- [ ] Text appears incrementally
- [ ] Shows cursor/typing indicator
- [ ] Handles completion smoothly
- [ ] Graceful error handling

---

## FE-011: Outfit Cards in Chat

**Phase:** 4 | **Priority:** P0 | **Dependencies:** FE-009, FE-006

**Description:**
Display outfit recommendations inline in chat messages.

**Files to Create:**
- `frontend/src/components/chat/OutfitSuggestion.tsx` - Inline outfit card

**Key Requirements:**
- Compact outfit card design
- Shows garment thumbnails
- Like/dislike buttons inline
- "Show more" for details
- Multiple outfits per message

**Acceptance Criteria:**
- [ ] Outfits display in chat
- [ ] Compact but readable
- [ ] Feedback buttons work
- [ ] Can expand for details

---

## FE-012: Preferences/Settings Page

**Phase:** 5 | **Priority:** P1 | **Dependencies:** BE-021

**Description:**
User preferences page for style settings.

**Files to Create:**
- `frontend/src/app/settings/page.tsx` - Settings page
- `frontend/src/components/settings/StylePreferences.tsx` - Style section
- `frontend/src/components/settings/ColorPreferences.tsx` - Color section

**Settings:**
- Preferred styles (multi-select)
- Avoid styles (multi-select)
- Favorite colors
- Avoid colors
- Preferred fit
- Formality range (slider)
- Comfort vs style balance (slider)

**Acceptance Criteria:**
- [ ] All preferences editable
- [ ] Changes saved to backend
- [ ] Form validation
- [ ] Success feedback

---

## FE-013: Outfit History View

**Phase:** 5 | **Priority:** P2 | **Dependencies:** BE-019

**Description:**
View past outfit recommendations and what user wore.

**Files to Create:**
- `frontend/src/app/outfits/history/page.tsx` - History page
- `frontend/src/components/outfits/HistoryList.tsx` - List component

**Key Requirements:**
- List of past outfits
- Filter by date, occasion
- Show feedback given
- Mark what was actually worn
- Calendar view (optional)

**Acceptance Criteria:**
- [ ] Shows past outfits
- [ ] Can filter by date
- [ ] Shows feedback status
- [ ] Paginated or scrollable

---

## FE-014: Mobile Responsive Design

**Phase:** 5 | **Priority:** P1 | **Dependencies:** All FE

**Description:**
Ensure all pages work well on mobile devices.

**Files to Modify:**
- All component files - Add responsive styles

**Key Requirements:**
- Responsive breakpoints (sm, md, lg, xl)
- Touch-friendly buttons and inputs
- Mobile navigation (hamburger menu or tabs)
- Swipeable outfit cards (optional)
- Optimized image sizes

**Acceptance Criteria:**
- [ ] Works on 320px width
- [ ] Touch targets >= 44px
- [ ] Navigation works on mobile
- [ ] No horizontal scroll
- [ ] Performance acceptable on mobile

