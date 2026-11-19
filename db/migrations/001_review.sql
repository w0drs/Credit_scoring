ALTER TABLE applications_database
    ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending';
