CREATE TABLE applications_database (
	id BIGINT,
	application_dt TEXT,
	education_cd TEXT,
	gender_cd TEXT,
	age BIGINT,
	car_own_flg TEXT,
	car_type_flg TEXT,
	appl_rej_cnt BIGINT,
	good_work_flg BIGINT,
	"Score_bki" FLOAT,
	out_request_cnt BIGINT,
	region_rating BIGINT,
	home_address_cd BIGINT,
	work_address_cd BIGINT,
	income BIGINT,
	"SNA" BIGINT,
	first_time_cd BIGINT,
	"Air_flg" TEXT
, review_status TEXT NOT NULL DEFAULT 'pending');
CREATE INDEX idx_applications_review
    ON applications_database (id, review_status);
CREATE INDEX idx_applications_review_status
    ON applications_database (review_status);
