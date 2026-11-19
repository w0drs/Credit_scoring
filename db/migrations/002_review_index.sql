-- depends: 001_review
CREATE INDEX idx_applications_review
    ON applications_database (id, review_status);
