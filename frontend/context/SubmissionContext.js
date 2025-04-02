"use client";

import React, { createContext, useReducer, useContext, useEffect } from "react";

// 1. Create the context
const SubmissionContext = createContext(null);

// 2. Define initial state for all sections
const initialState = {
	authors: [],
	details: {
		title: "",
		abstract: "",
	},
	keywords: [],
	documents: [],
	reviewers: [],
	letter: "",
	// Add any other fields as needed
};

// 3. Create the reducer function
function submissionReducer(state, action) {
	switch (action.type) {
		/* ---------------------------
		   Authors
	   --------------------------- */
		case "ADD_AUTHOR":
			return {
				...state,
				authors: [...state.authors, action.payload],
			};
		case "UPDATE_AUTHOR":
			return {
				...state,
				authors: state.authors.map((author, idx) =>
					idx === action.payload.index ? action.payload.author : author
				),
			};
		case "REMOVE_AUTHOR":
			return {
				...state,
				authors: state.authors.filter((_, idx) => idx !== action.payload),
			};

		/* ---------------------------
		   Details
		--------------------------- */
		case "UPDATE_DETAILS":
			return {
				...state,
				details: {
					...state.details,
					...action.payload, // e.g., { title, abstract }
				},
			};

		/* ---------------------------
		   Keywords
		--------------------------- */
		case "ADD_KEYWORD":
			return {
				...state,
				keywords: [...state.keywords, action.payload],
			};
		case "REMOVE_KEYWORD":
			return {
				...state,
				keywords: state.keywords.filter((_, idx) => idx !== action.payload),
			};

		/* ---------------------------
		   Documents
		--------------------------- */
		case "ADD_DOCUMENT":
			return {
				...state,
				documents: [...state.documents, action.payload],
			};
		case "REMOVE_DOCUMENT":
			return {
				...state,
				documents: state.documents.filter((_, idx) => idx !== action.payload),
			};

		/* ---------------------------
		   Reviewers
		--------------------------- */
		case "ADD_REVIEWER":
			return {
				...state,
				reviewers: [...state.reviewers, action.payload],
			};
		case "REMOVE_REVIEWER":
			return {
				...state,
				reviewers: state.reviewers.filter((_, idx) => idx !== action.payload),
			};

		/* ---------------------------
		   Letter
		--------------------------- */
		case "UPDATE_LETTER":
			return {
				...state,
				letter: action.payload,
			};
			
		/* ---------------------------
		   Reset
		--------------------------- */
		case "RESET":
			return initialState;

		/* ---------------------------
		   Default
		--------------------------- */
		default:
			return state;
	}
}

// 4. Create the Provider Component
export function SubmissionProvider({ children }) {
	// On initial load, try to read from localStorage
	const [state, dispatch] = useReducer(submissionReducer, initialState, (init) => {
		if (typeof window !== "undefined") {
			const savedData = localStorage.getItem("submissionData");
			return savedData ? JSON.parse(savedData) : init;
		}
		return init;
	});

	// Whenever state changes, save to localStorage
	useEffect(() => {
		if (typeof window !== "undefined") {
			localStorage.setItem("submissionData", JSON.stringify(state));
		}
	}, [state]);

	return (
		<SubmissionContext.Provider value={ { state, dispatch } }>
			{ children }
		</SubmissionContext.Provider>
	);
}

// 5. Create a custom hook to easily use context
export function useSubmissionContext() {
	const context = useContext(SubmissionContext);
	if (!context) {
		throw new Error("useSubmissionContext must be used within a SubmissionProvider");
	}
	return context;
}
